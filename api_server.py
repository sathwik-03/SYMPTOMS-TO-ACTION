"""
FastAPI backend for the Medical Reasoning System.
Exposes the diagnosis pipeline as REST endpoints.
"""

import sys
import os
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(title="Medical Reasoning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────
sessions: Dict[str, dict] = {}

# ── Request / Response models ──────────────────────────────

class StartRequest(BaseModel):
    symptoms: List[str]
    alpha: float = 0.4
    beta: float = 0.6

class AnswerRequest(BaseModel):
    session_id: str
    answers: Dict[str, str]  # symptom -> "yes" | "no"

# ── Lazy-loaded pipeline singletons ────────────────────────
_pipeline_ready = False

def _ensure_pipeline():
    """Import heavy modules once on first request."""
    global _pipeline_ready
    if not _pipeline_ready:
        # These imports trigger model loading
        from main_pipeline import diagnose          # noqa: F401
        from tree_of_thoughts import get_tot        # noqa: F401
        _pipeline_ready = True

# ── Endpoints ──────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/diagnose/start")
def start_diagnosis(req: StartRequest):
    """
    Start a new diagnostic session.
    Returns initial ranked diseases + first round of follow-up questions.
    """
    _ensure_pipeline()
    from main_pipeline import diagnose
    from tree_of_thoughts import get_tot

    if not req.symptoms:
        raise HTTPException(400, "At least one symptom is required")

    # Run initial diagnosis
    result = diagnose(req.symptoms, alpha=req.alpha, beta=req.beta)

    tot = get_tot()
    tree = result.get("tree", [])

    # Get first round of questions
    questions = tot.get_next_questions(tree, max_questions=3)

    # Create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "symptoms": list(req.symptoms),
        "tree": tree,
        "iteration": 1,
        "all_answers": {},
        "result": result,
    }

    return {
        "session_id": session_id,
        "diagnoses": result["diagnoses"][:5],
        "confidence": result["confidence"],
        "questions": questions,
        "iteration": 1,
        "is_final": False,
    }


@app.post("/api/diagnose/answer")
def answer_questions(req: AnswerRequest):
    """
    Submit answers to follow-up questions and get updated results.
    """
    _ensure_pipeline()
    from main_pipeline import diagnose
    from tree_of_thoughts import get_tot

    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    tot = get_tot()
    tree = session["tree"]
    current_symptoms = list(session["symptoms"])

    # Record answers
    session["all_answers"].update(req.answers)

    # Add confirmed symptoms to the symptom list
    for symptom, answer in req.answers.items():
        if answer.lower() in ["yes", "y"]:
            symptom_readable = symptom.replace("_", " ").lower()
            if symptom_readable not in [s.lower() for s in current_symptoms]:
                current_symptoms.append(symptom_readable)

    # Update the tree with answers
    tree = tot.update_tree_with_answers(tree, req.answers)

    # Check for final diagnosis
    final = tot.get_final_diagnosis(tree, confidence_threshold=0.7)
    iteration = session["iteration"] + 1

    # Re-run diagnosis with expanded symptoms
    result = diagnose(current_symptoms, alpha=0.4, beta=0.6)

    # Get next questions
    questions = tot.get_next_questions(tree, max_questions=3)

    is_final = final is not None or iteration > 5 or len(questions) == 0

    # Update session
    session["symptoms"] = current_symptoms
    session["tree"] = tree
    session["iteration"] = iteration
    session["result"] = result

    response = {
        "session_id": req.session_id,
        "diagnoses": result["diagnoses"][:5],
        "confidence": result["confidence"],
        "questions": questions if not is_final else [],
        "iteration": iteration,
        "is_final": is_final,
    }

    if final:
        response["final_diagnosis"] = final

    # Clean up if done
    if is_final and req.session_id in sessions:
        del sessions[req.session_id]

    return response


@app.get("/api/symptoms/common")
def common_symptoms():
    """Return a list of common symptoms for the autocomplete UI."""
    common = [
        "high fever", "cough", "headache", "chest pain", "fatigue",
        "vomiting", "nausea", "stomach pain", "dizziness", "shortness of breath",
        "skin rash", "joint pain", "back pain", "sore throat", "runny nose",
        "muscle pain", "diarrhea", "constipation", "weight loss", "sweating",
        "chills", "blurred vision", "swelling", "itching", "burning sensation",
        "loss of appetite", "abdominal pain", "anxiety", "depression",
        "palpitations", "chest tightness", "body aches", "cold hands",
        "dark urine", "yellowing of eyes", "excessive hunger", "frequent urination",
    ]
    return {"symptoms": common}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
