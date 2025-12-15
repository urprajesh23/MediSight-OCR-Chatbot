# app.py -- Robust Flask app for PaddleOCR + optional Gemini (backend key from .env)
# Put this file in your project root (same folder as templates/ and static/).
# Usage:
#   - create .env with GEMINI_API_KEY=your_key (optional)
#   - activate venv and run `python app.py`

import os
import re
import json
import time
import traceback
from pathlib import Path

# Flask imports
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, jsonify
)
from werkzeug.utils import secure_filename

# Load .env (if present)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    # dotenv is optional; we just continue
    pass

# ------------------------
# Environment and config
# ------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
MAX_CONTENT_MB = int(os.environ.get("MAX_CONTENT_MB", 40))
MAX_CONTENT_BYTES = MAX_CONTENT_MB * 1024 * 1024

# ------------------------
# NumPy compatibility shim (for PaddleOCR/PaddlePaddle older expectations)
# ------------------------
try:
    import numpy as np

    if not hasattr(np, "int"): np.int = int
    if not hasattr(np, "bool"): np.bool = bool
    if not hasattr(np, "float"): np.float = float
    if not hasattr(np, "object"): np.object = object
    if not hasattr(np, "long"): np.long = int
    if not hasattr(np, "complex"): np.complex = complex
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int64],
            "uint": [np.uint64],
            "float": [np.float64],
            "complex": [np.complex128],
            "others": [],
        }
except Exception as e:
    print("Warning: numpy import/shim failed:", e)

# ------------------------
# PaddleOCR init (safe)
# ------------------------
try:
    from paddleocr import PaddleOCR
    # Use modern option; PaddleOCR will default to CPU if GPU not available
    OCR_INSTANCE = PaddleOCR(use_textline_orientation=True, lang='en')
    print("PaddleOCR initialized.")
except Exception as e:
    OCR_INSTANCE = None
    print("Error initializing PaddleOCR. OCR functionality will be unavailable. Error:", e)
    traceback.print_exc()

# ------------------------
# Optional Gemini (Google GenAI) SDK import
# ------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or None
genai = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai_pkg
        genai = genai_pkg
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            print("Gemini (google.generativeai) configured from environment.")
        except Exception as e:
            print("Warning: genai.configure failed:", e)
    except Exception as e:
        genai = None
        print("Gemini library (google.generativeai) not available or failed to import:", e)
else:
    # Not configured; fine — Gemini optional
    pass

# ------------------------
# Flask app
# ------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES

# ------------------------
# Utility helpers
# ------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Robust OCR wrapper that handles PaddleOCR old/new APIs and returns cleaned text
def run_ocr(path: str) -> str:
    if OCR_INSTANCE is None:
        raise RuntimeError("PaddleOCR is not available in this Python environment.")

    try:
        # Prefer modern predict() if present
        if hasattr(OCR_INSTANCE, "predict"):
            res = OCR_INSTANCE.predict(path)
        else:
            # fallback to older ocr()
            res = OCR_INSTANCE.ocr(path)
    except TypeError:
        # signature mismatch; try fallback
        res = OCR_INSTANCE.ocr(path)
    except Exception:
        # re-raise upwards so caller logs traceback
        raise

    # Walk result recursively and collect plausible text strings
    texts = []
    def walk(item):
        if item is None:
            return
        if isinstance(item, str):
            texts.append(item)
            return
        if isinstance(item, (list, tuple)):
            for it in item:
                walk(it)
            return
        if isinstance(item, dict):
            for v in item.values():
                walk(v)
            return
        # fallback - small reprs only
        try:
            s = str(item)
            if 0 < len(s) <= 300 and any(c.isalpha() for c in s):
                texts.append(s)
        except Exception:
            pass

    walk(res)

    # Clean lines: remove file paths, python object reprs, pure punctuation
    cleaned = []
    seen = set()
    for t in texts:
        line = t.strip()
        if not line:
            continue
        if not re.search(r"[A-Za-z]", line):
            continue
        # skip common non-text remnants
        if re.search(r"[A-Za-z]:\\|/static/uploads/|/home/|C:\\\\", line):
            continue
        if re.search(r"<paddlex\.|object at 0x|0x[0-9A-Fa-f]{6,}", line):
            continue
        if len(line) < 2:
            continue
        if line in seen:
            continue
        seen.add(line)
        cleaned.append(line)
    return "\n".join(cleaned)

# Candidate builder (n-grams) with filters
STOPWORDS = {
    "medicine","medicines","drug","drugs","injection","injections",
    "tablet","tablets","cap","capsule","capsules","mg","ml","mcg",
    "strip","pack","box","syrup","g","oral","dose","dosage","ip",
    "usp","expiry","mfg","manufacturer","each","contains","for","use"
}

def build_candidates(text: str, max_n: int = 3):
    raw = [t.strip() for t in re.split(r"[,\n\r\t]+", text) if t.strip()]
    tokens = []
    for r in raw:
        tokens.extend(r.split())

    ngrams = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ng = " ".join(tokens[i:i+n])
            words = ng.lower().split()
            if all(w in STOPWORDS for w in words): continue
            if not any(re.search(r"[a-zA-Z]", w) for w in words): continue
            # filters
            if re.search(r"[A-Za-z]:\\|/static/uploads/|/home/|C:\\\\", ng): continue
            if re.search(r"object at 0x|<paddlex|0x[0-9A-Fa-f]{6,}", ng): continue
            if len(ng) <= 1: continue
            if ng not in ngrams:
                ngrams.append(ng)
    return ngrams

# Gemini helpers (optional; safe if genai is None)
def extract_with_gemini(text: str):
    if not genai:
        return []
    try:
        model = genai.GenerativeModel("models/gemini-flash-lite-latest")
        prompt = (
            "Extract only medicine product names from OCR text.\n"
            "Ignore generic words like medicine/tablet/mg/ml.\n"
            "Return ONLY JSON array: [{\"name\":\"...\",\"score\":int,\"matched_span\":\"...\"}]\n\n"
            "OCR TEXT:\n" + text
        )
        resp = model.generate_content(prompt)
        out = resp.text.strip()
        m = re.search(r"(\[.*\])", out, re.S)
        if not m:
            return []
        lst = json.loads(m.group(1))
        for item in lst:
            if "name" in item:
                item["name"] = re.sub(r"\s+", " ", item["name"]).strip()
        return lst
    except Exception as e:
        print("Gemini extractor error:", e)
        traceback.print_exc()
        return []

def final_answer(name: str):
    if not genai:
        return "Gemini not configured in backend."
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
Detected medicine: {name}

1) What is this medicine?
2) What does it treat?
3) How long to show results?
4) When to take (before/after food)?
5) Precautions & side effects?

Answer concisely.
"""
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print("Gemini final answer error:", e)
        traceback.print_exc()
        return f"Error calling Gemini: {e}"

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Debugging visibility
    print(">>> DEBUG: request.form keys:", list(request.form.keys()))
    print(">>> DEBUG: request.files keys:", list(request.files.keys()))

    # Validate file presence
    if "image" not in request.files:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": "no-file"}), 400
        flash("No file part", "danger")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": "empty-filename"}), 400
        flash("No selected file", "danger")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": "unsupported-type"}), 400
        flash("Unsupported file type.", "danger")
        return redirect(url_for("index"))

    # Save file safely
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    save_name = f"{timestamp}_{filename}"
    save_path = UPLOAD_FOLDER / save_name
    try:
        file.save(str(save_path))
    except Exception as e:
        print(">>> ERROR saving file:", e)
        traceback.print_exc()
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": "save-failed", "detail": str(e)}), 500
        flash("Error saving file.", "danger")
        return redirect(url_for("index"))

    # Run OCR
    try:
        ocr_text = run_ocr(str(save_path))
    except Exception as e:
        tb = traceback.format_exc()
        print(">>> OCR error (traceback):")
        print(tb)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": "ocr-failed", "detail": str(e), "traceback": tb}), 500
        flash("OCR error: " + str(e), "danger")
        return redirect(url_for("index"))

    # Build local candidates
    local_cands = build_candidates(ocr_text)

    # Gemini extraction (only if configured)
    gemini_cands = []
    chosen = None
    final = None
    if genai and GEMINI_API_KEY:
        try:
            gemini_cands = extract_with_gemini(ocr_text)
            if gemini_cands:
                top = max(gemini_cands, key=lambda x: x.get("score", 0))
                chosen = top
                if chosen.get("name"):
                    final = final_answer(chosen["name"])
        except Exception as e:
            print("Gemini processing error:", e)
            traceback.print_exc()
            gemini_cands = []

    # Save results JSON alongside image
    results_obj = {
        "ocr_text": ocr_text,
        "local_cands": local_cands,
        "gemini_cands": gemini_cands,
        "chosen": chosen,
        "final": final,
    }
    json_path = UPLOAD_FOLDER / (save_name + ".json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(">>> ERROR saving results json:", e)
        traceback.print_exc()

    # Respond for AJAX clients with JSON filename
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"ok": True, "filename": save_name})

    # Fallback: render results immediately
    return render_template("results.html",
                           image_url=url_for("uploaded_file", filename=save_name),
                           ocr_text=ocr_text,
                           local_cands=local_cands,
                           gemini_cands=gemini_cands,
                           chosen=chosen,
                           final=final)

@app.route("/results", methods=["GET"])
def results():
    img = request.args.get("img")
    if not img:
        flash("Missing image parameter", "danger")
        return redirect(url_for("index"))

    image_path = UPLOAD_FOLDER / img
    json_path = UPLOAD_FOLDER / (img + ".json")

    if not image_path.exists():
        flash("Image not found", "danger")
        return redirect(url_for("index"))

    data = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(">>> ERROR reading json:", e)
            traceback.print_exc()

    ocr_text = data.get("ocr_text", "")
    local_cands = data.get("local_cands", [])
    gemini_cands = data.get("gemini_cands", [])
    chosen = data.get("chosen")
    final = data.get("final")

    if not ocr_text:
        try:
            ocr_text = run_ocr(str(image_path))
            local_cands = build_candidates(ocr_text)
        except Exception as e:
            print("OCR error in results view:", e)
            traceback.print_exc()
            flash("OCR error on results view: " + str(e), "danger")
            return redirect(url_for("index"))

    return render_template("results.html",
                           image_url=url_for("uploaded_file", filename=img),
                           ocr_text=ocr_text,
                           local_cands=local_cands,
                           gemini_cands=gemini_cands,
                           chosen=chosen,
                           final=final)

@app.route("/ask", methods=["POST"])
def ask():
    if not genai or not GEMINI_API_KEY:
        return jsonify({"ok": False, "error": "Gemini not configured on server."}), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    name = (data.get("name") or "").strip()
    ocr_text = (data.get("ocr_text") or "").strip()

    if not question:
        return jsonify({"ok": False, "error": "Missing question"}), 400

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt_parts = [
        "You are a helpful medicine assistant.",
        "You may give GENERAL and COMMON medical guidance based on widely accepted usage.",
        "If exact dosage or timing is not available, give SAFE GENERAL ADVICE.",
        "Answer in 4–6 bullet points.",
        "Always add a short safety disclaimer.",
        "Do NOT hallucinate exact dosages or medical claims.",
        ]


        if name:
            prompt_parts.append(f"Medicine name: {name}")

        prompt_parts.append(f"User question: {question}")

        if ocr_text:
            prompt_parts.append(
                f"OCR text from medicine strip or box (may be incomplete):\n{ocr_text[:3000]}"
            )

        prompt = "\n\n".join(prompt_parts)

        resp = model.generate_content(prompt)
        answer = (resp.text or "").strip()

        return jsonify({"ok": True, "answer": answer})

    except Exception as e:
        print("Gemini chat error:", e)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ------------------------
# Run server
# ------------------------
if __name__ == "__main__":
    try:
        print(f"Starting Flask app. Upload folder: {app.config['UPLOAD_FOLDER']}")
        if GEMINI_API_KEY:
            print("GEMINI_API_KEY is set (will attempt Gemini calls).")
        else:
            print("GEMINI_API_KEY not set — Gemini features disabled.")
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    except Exception:
        print("Fatal error running Flask app:")
        traceback.print_exc()
