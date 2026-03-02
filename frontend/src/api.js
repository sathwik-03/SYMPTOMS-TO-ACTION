const API_BASE = "http://localhost:8000";

export async function startDiagnosis(symptoms) {
    const res = await fetch(`${API_BASE}/api/diagnose/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Server error" }));
        throw new Error(err.detail || "Failed to start diagnosis");
    }
    return res.json();
}

export async function submitAnswers(sessionId, answers) {
    const res = await fetch(`${API_BASE}/api/diagnose/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, answers }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Server error" }));
        throw new Error(err.detail || "Failed to submit answers");
    }
    return res.json();
}

export async function getCommonSymptoms() {
    const res = await fetch(`${API_BASE}/api/symptoms/common`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.symptoms || [];
}
