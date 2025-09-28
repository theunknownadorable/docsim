from flask import Flask, render_template, request, jsonify, session, send_file, make_response
import os, json, difflib
from datetime import datetime
import google.generativeai as genai
import pdfkit
import base64
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("DOCSIM_SECRET_KEY", "docsim-dev-secret")

# Configure Gemini (read API key from environment variable for safety)
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    # No key set; genai.configure with empty string will still let the app run but API calls will fail.
    genai.configure(api_key="")
else:
    genai.configure(api_key=GEMINI_KEY)

# Preferred model (fallback handled if unavailable)
MODEL_NAME = "gemini-2.5-flash"

def safe_generate(prompt, instruction=None):
    """
    Wrapper around gemini generate API. Returns text on success or raises Exception.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Use generate_content if available
        resp = model.generate_content(prompt)
        # resp can be string-like or object; try to extract .text
        if hasattr(resp, "text"):
            return resp.text
        # fallback to str(resp)
        return str(resp)
    except Exception as e:
        # Bubble up error to caller
        raise

def make_dummy_case(case_id=1):
    """Return a deterministic dummy case (used when API fails)."""
    return {
        "id": case_id,
        "patient_history": "45-year-old male with 3 days of fever, cough, and shortness of breath.",
        "symptoms": "Fever, productive cough, increasing dyspnea on exertion.",
        "vital_signs": {"heart_rate": "98 bpm", "bp": "130/80 mmHg", "resp_rate": "22/min", "temp": "38.4 C", "oxygen_saturation": "92%"},
        "available_tests": ["Chest X-Ray", "CBC", "Blood Culture", "COVID PCR"],
        # map test name -> result
        "test_results": {
            "Chest X-Ray": "Bilateral patchy infiltrates, more pronounced in lower lobes.",
            "CBC": "WBC 12.5 x10^9/L, neutrophil predominance.",
            "Blood Culture": "No growth at 48 hours.",
            "COVID PCR": "Negative"
        },
        "correct_diagnosis": "Community-acquired pneumonia",
        "differential_diagnoses": ["Viral bronchitis", "COVID-19", "Pulmonary embolism"],
        "explanation": "Presentation and x-ray findings suggest bacterial pneumonia; tachypnea and hypoxia support this; blood tests show leukocytosis."
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulator')
def simulator():
    return render_template('simulator.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

# --- API endpoints expected by UI ---

@app.route('/api/new_case', methods=['GET'])
def api_new_case():
    """
    Generate a new case using Gemini. The UI expects a JSON object with fields:
    id, patient_history, symptoms, vital_signs (dict), available_tests (list), test_results (map),
    correct_diagnosis, differential_diagnoses, explanation
    """
    # Try to call Gemini to produce structured JSON
    prompt = (
        "Generate a realistic medical patient case as JSON with the following keys:\n"
        "patient_history, symptoms, vital_signs (object), available_tests (array of test names), "
        "test_results (object mapping test name to a short result), correct_diagnosis (string), "
        "differential_diagnoses (array), explanation (string).\n\n"
        "Keep JSON parsable and avoid extra commentary. Use concise values.\n"
    )
    try:
        text = safe_generate(prompt)
        # Attempt to locate first '{' and parse JSON substring
        start = text.find('{')
        if start != -1:
            jtext = text[start:]
            try:
                data = json.loads(jtext)
            except Exception:
                # Sometimes the model outputs with trailing text; try to extract JSON by simple heuristics
                end = jtext.rfind('}')
                if end != -1:
                    jtext2 = jtext[:end+1]
                    data = json.loads(jtext2)
                else:
                    raise
        else:
            raise ValueError("No JSON produced by model")
        # Enrich and set an id
        data.setdefault('id', int(datetime.utcnow().timestamp()))
        # Make sure keys exist
        data.setdefault('vital_signs', {})
        data.setdefault('available_tests', list(data.get('test_results', {}).keys()) or [])
        # store current_case in session for ordering tests and evaluation
        session['current_case'] = data
        return jsonify(data)
    except Exception as e:
        # On error, return a dummy case so UI still works
        dummy = make_dummy_case(case_id=int(datetime.utcnow().timestamp()))
        session['current_case'] = dummy
        return jsonify(dummy)

@app.route('/api/order_test', methods=['POST'])
def api_order_test():
    """
    Body: { "test_name": "<name>" }
    Returns: { "test_name": ..., "result": "..." }
    """
    payload = request.get_json(force=True)
    test_name = payload.get('test_name')
    current = session.get('current_case')
    if not current:
        return jsonify({"error": "No active case"}), 400
    results = current.get('test_results', {})
    result = results.get(test_name, "No specific result available for this test.")
    # track ordered tests
    ordered = session.get('ordered_tests', [])
    ordered.append({"test_name": test_name, "result": result})
    session['ordered_tests'] = ordered
    session.modified = True
    return jsonify({"test_name": test_name, "result": result})

@app.route('/api/submit_diagnosis', methods=['POST'])
def api_submit_diagnosis():
    """
    Body: { "diagnosis": "<text>" }
    Returns feedback object expected by UI:
    {
      "is_correct": bool,
      "submitted_diagnosis": str,
      "correct_diagnosis": str,
      "similarity_score": float,
      "explanation": str,
      "consequences": str,
      "differential_diagnoses": [...]
    }
    """
    payload = request.get_json(force=True)
    diag = (payload.get('diagnosis') or "").strip()
    current = session.get('current_case')
    if not current:
        return jsonify({"error": "No active case"}), 400

    correct = current.get('correct_diagnosis') or ""
    # similarity check (case-insensitive)
    sim = difflib.SequenceMatcher(None, diag.lower(), (correct or "").lower()).ratio()
    is_correct = sim >= 0.6  # threshold: 60% similarity

    # If Gemini is available, ask it to produce a richer evaluation
    try:
        eval_prompt = (
            f"Case: {json.dumps(current)}\n\n"
            f"Student diagnosis: {diag}\n\n"
            "Provide a JSON object with keys: correct_diagnosis, differential_diagnoses (array), "
            "explanation (string), consequences (string). Keep it concise and parsable."
        )
        eval_text = safe_generate(eval_prompt)
        start = eval_text.find('{')
        if start != -1:
            jtext = eval_text[start:]
            end = jtext.rfind('}')
            if end != -1:
                jtext = jtext[:end+1]
            eval_json = json.loads(jtext)
        else:
            raise ValueError("No JSON from model")
        correct_from_model = eval_json.get('correct_diagnosis', correct)
        differential = eval_json.get('differential_diagnoses', current.get('differential_diagnoses', []))
        explanation = eval_json.get('explanation', current.get('explanation', ''))
        consequences = eval_json.get('consequences', '')
    except Exception:
        # fallback to using case fields
        correct_from_model = correct
        differential = current.get('differential_diagnoses', [])
        explanation = current.get('explanation', '')
        consequences = "Misdiagnosis can lead to delayed treatment or inappropriate therapy."

    feedback = {
        "is_correct": is_correct,
        "submitted_diagnosis": diag,
        "correct_diagnosis": correct_from_model,
        "similarity_score": sim,
        "explanation": explanation,
        "consequences": consequences,
        "differential_diagnoses": differential
    }

    # append to history stored in session
    history = session.get('diagnosis_history', [])
    history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "case": current,
        "submitted_diagnosis": diag,
        "feedback": feedback
    })
    session['diagnosis_history'] = history
    # clear current case so user will load a fresh one next time (optional)
    session.pop('current_case', None)
    session.pop('ordered_tests', None)
    session.modified = True

    return jsonify(feedback)

@app.route('/api/history', methods=['GET'])
def api_history():
    history = session.get('diagnosis_history', [])
    cases_completed = len(history)
    correct = sum(1 for h in history if h['feedback']['is_correct'])
    accuracy = (correct / cases_completed * 100) if cases_completed > 0 else 0

    # Compute best streak
    streak, best_streak = 0, 0
    for h in history:
        if h['feedback']['is_correct']:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0

    return jsonify({
        "cases_completed": cases_completed,
        "accuracy": round(accuracy, 2),
        "best_streak": best_streak,
        "history": history
    })

# --- Chatbot endpoint ---
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(force=True)
    question = data.get('message', '').strip()
    if not question:
        return jsonify({"reply": "Please ask a question."})

    try:
        prompt = f"""You are a professional, evidence-based medical assistant. Answer briefly and clearly.
        Question: {question}
        """

        resp = safe_generate(prompt)
        # return raw text as reply
        return jsonify({"reply": resp})
    except Exception as e:
        return jsonify({"reply": "Chat service currently unavailable. (No API key configured or model error.)"})

# Helper to download a session report as plain text
@app.route('/download_report')
def download_report_pdf():
    history = session.get('diagnosis_history', [])
    doctor = session.get('doctor_name', 'Unknown')

    # Generate charts as base64 (you can send chart data to template)
    correct = sum(1 for h in history if h['feedback']['is_correct'])
    wrong = len(history) - correct

    accuracy_over_time = []
    correct_count = 0
    date_labels = []
    for i, h in enumerate(history):
        if h['feedback']['is_correct']:
            correct_count += 1
        accuracy_over_time.append(round(correct_count / (i+1) * 100))
        # Pre-process date labels to avoid using split filter in template
        date_labels.append(h['timestamp'].split('T')[0])

    # Render HTML template with embedded charts (using Chart.js inline script)
    rendered = render_template("report_pdf.html",
                               history=history,
                               doctor=doctor,
                               correct=correct,
                               wrong=wrong,
                               accuracy_over_time=accuracy_over_time,
                               date_labels=date_labels)
    # Convert HTML to PDF
    pdf = pdfkit.from_string(rendered, False)

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=docsim_report.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)
