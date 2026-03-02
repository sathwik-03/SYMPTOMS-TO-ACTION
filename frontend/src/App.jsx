import { useState, useRef, useEffect, useCallback } from "react";
import "./index.css";
import { startDiagnosis, submitAnswers, getCommonSymptoms } from "./api";

/* ── Helpers ─────────────────────────────────────────── */
const FALLBACK_SUGGESTIONS = [
  "high fever", "cough", "headache", "chest pain", "fatigue",
  "vomiting", "nausea", "stomach pain", "dizziness",
  "shortness of breath", "skin rash", "joint pain",
  "sore throat", "runny nose", "muscle pain", "diarrhea",
];

function confidenceColor(c) {
  if (c >= 0.7) return "high";
  if (c >= 0.4) return "medium";
  return "low";
}

function formatPct(n) {
  return `${(n * 100).toFixed(1)}%`;
}

/* ── Main App  ───────────────────────────────────────── */
export default function App() {
  const [phase, setPhase] = useState("intake"); // intake | loading | questioning | result
  const [symptoms, setSymptoms] = useState([]);
  const [inputVal, setInputVal] = useState("");
  const [suggestions, setSuggestions] = useState(FALLBACK_SUGGESTIONS);

  // Diagnosis state
  const [sessionId, setSessionId] = useState(null);
  const [diagnoses, setDiagnoses] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [confidence, setConfidence] = useState("low");
  const [iteration, setIteration] = useState(0);
  const [chatLog, setChatLog] = useState([]);
  const [finalDiag, setFinalDiag] = useState(null);
  const [error, setError] = useState(null);
  const [pendingAnswers, setPendingAnswers] = useState({});

  const chatEndRef = useRef(null);

  // Fetch common symptoms on mount
  useEffect(() => {
    getCommonSymptoms()
      .then((s) => s.length > 0 && setSuggestions(s))
      .catch(() => { });
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatLog, questions]);

  /* ── Symptom input handlers ────────────────────────── */
  const addSymptom = useCallback(
    (s) => {
      const clean = s.trim().toLowerCase();
      if (clean && !symptoms.includes(clean)) {
        setSymptoms((prev) => [...prev, clean]);
      }
      setInputVal("");
    },
    [symptoms]
  );

  const removeSymptom = (s) => setSymptoms((prev) => prev.filter((x) => x !== s));

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && inputVal.trim()) {
      addSymptom(inputVal);
    }
  };

  /* ── Start Diagnosis ───────────────────────────────── */
  const handleStart = async () => {
    if (symptoms.length === 0) return;
    setPhase("loading");
    setError(null);
    setChatLog([{ type: "system", text: `Starting diagnosis with: ${symptoms.join(", ")}` }]);

    try {
      const res = await startDiagnosis(symptoms);
      setSessionId(res.session_id);
      setDiagnoses(res.diagnoses || []);
      setQuestions(res.questions || []);
      setConfidence(res.confidence);
      setIteration(res.iteration);

      setChatLog((prev) => [
        ...prev,
        {
          type: "system",
          text: `Analyzed ${symptoms.length} symptoms. Found ${(res.diagnoses || []).length} possible conditions.\nLet me ask some follow-up questions to narrow it down.`,
        },
      ]);

      if (res.is_final || (res.questions || []).length === 0) {
        setFinalDiag(res.diagnoses?.[0] || null);
        setPhase("result");
      } else {
        setPhase("questioning");
      }
    } catch (err) {
      setError(err.message);
      setPhase("intake");
    }
  };

  /* ── Answer a question ─────────────────────────────── */
  const handleAnswer = (symptom, answer) => {
    setPendingAnswers((prev) => ({ ...prev, [symptom]: answer }));

    setChatLog((prev) => [
      ...prev,
      {
        type: "system",
        text: `Do you have ${symptom.replace(/_/g, " ").toLowerCase()}?`,
      },
      { type: "user", text: answer === "yes" ? "✅ Yes" : "❌ No" },
    ]);

    // Remove this question from the list
    setQuestions((prev) => prev.filter((q) => q.symptom !== symptom));
  };

  /* ── Submit batch of answers ───────────────────────── */
  const handleSubmitAnswers = async () => {
    if (Object.keys(pendingAnswers).length === 0) return;
    setPhase("loading");

    try {
      const res = await submitAnswers(sessionId, pendingAnswers);
      setDiagnoses(res.diagnoses || []);
      setQuestions(res.questions || []);
      setConfidence(res.confidence);
      setIteration(res.iteration);
      setPendingAnswers({});

      if (res.is_final) {
        setFinalDiag(res.final_diagnosis || res.diagnoses?.[0] || null);
        setChatLog((prev) => [
          ...prev,
          { type: "system", text: "✨ Analysis complete! Here are the results." },
        ]);
        setPhase("result");
      } else {
        setChatLog((prev) => [
          ...prev,
          {
            type: "system",
            text: `Updated analysis (Round ${res.iteration}). I have ${(res.questions || []).length} more questions.`,
          },
        ]);
        setPhase("questioning");
      }
    } catch (err) {
      setError(err.message);
      setPhase("questioning");
    }
  };

  /* ── Reset ─────────────────────────────────────────── */
  const handleReset = () => {
    setPhase("intake");
    setSymptoms([]);
    setInputVal("");
    setSessionId(null);
    setDiagnoses([]);
    setQuestions([]);
    setConfidence("low");
    setIteration(0);
    setChatLog([]);
    setFinalDiag(null);
    setError(null);
    setPendingAnswers({});
  };

  /* ── Render ────────────────────────────────────────── */
  return (
    <>
      <div className="bg-gradient-mesh" />
      <div className="app-container">
        {/* Header */}
        <header className="app-header">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <h1>MedReason AI</h1>
          </div>
          <p className="subtitle">
            Hybrid medical reasoning with Tree-of-Thoughts &amp; RAG-powered explainability
          </p>
        </header>

        {/* Error banner */}
        {error && (
          <div
            className="glass-card animate-fade-in"
            style={{
              padding: "1rem",
              marginBottom: "1.5rem",
              borderColor: "rgba(239,68,68,0.4)",
              maxWidth: 700,
              margin: "0 auto 1.5rem",
            }}
          >
            <span style={{ color: "var(--danger)" }}>⚠️ {error}</span>
          </div>
        )}

        {/* ── INTAKE PHASE ────────────────────────────── */}
        {phase === "intake" && (
          <section className="intake-section">
            <div className="glass-card" style={{ padding: "2rem" }}>
              <h2 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: "0.4rem" }}>
                What symptoms are you experiencing?
              </h2>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: "1.25rem" }}>
                Type a symptom and press Enter, or click the suggestions below.
              </p>

              <div className="input-wrapper">
                <input
                  id="symptom-input"
                  type="text"
                  placeholder="e.g. high fever, cough, chest pain…"
                  value={inputVal}
                  onChange={(e) => setInputVal(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
                <button
                  className="add-btn"
                  onClick={() => inputVal.trim() && addSymptom(inputVal)}
                  title="Add symptom"
                >
                  +
                </button>
              </div>

              {/* Tags */}
              <div className="symptom-tags">
                {symptoms.map((s) => (
                  <span key={s} className="symptom-tag">
                    {s}
                    <button onClick={() => removeSymptom(s)}>×</button>
                  </span>
                ))}
              </div>

              {/* Suggestions */}
              <div className="suggestions">
                <div className="suggestions-label">Common symptoms</div>
                <div className="suggestions-list">
                  {suggestions
                    .filter((s) => !symptoms.includes(s))
                    .slice(0, 12)
                    .map((s) => (
                      <button key={s} className="suggestion-chip" onClick={() => addSymptom(s)}>
                        {s}
                      </button>
                    ))}
                </div>
              </div>

              <button
                id="start-diagnosis-btn"
                className="btn-primary"
                onClick={handleStart}
                disabled={symptoms.length === 0}
              >
                {symptoms.length === 0
                  ? "Add at least one symptom"
                  : `Analyze ${symptoms.length} symptom${symptoms.length > 1 ? "s" : ""}`}
              </button>
            </div>
          </section>
        )}

        {/* ── LOADING PHASE ───────────────────────────── */}
        {phase === "loading" && (
          <div className="loading-container">
            <div className="loading-spinner" />
            <div className="loading-text">Analyzing symptoms with hybrid reasoning engine…</div>
          </div>
        )}

        {/* ── QUESTIONING PHASE ───────────────────────── */}
        {phase === "questioning" && (
          <div className="diagnosis-layout">
            {/* Left: Chat & Questions */}
            <div className="glass-card panel">
              <div className="panel-title">
                <span className="icon">💬</span>
                Follow-up Questions
                <span className="iteration-badge" style={{ marginLeft: "auto" }}>
                  Round {iteration}
                </span>
              </div>

              <div className="chat-messages">
                {chatLog.map((msg, i) => (
                  <div key={i} className={`chat-bubble ${msg.type}`}>
                    {msg.text}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>

              {/* Question cards */}
              {questions.length > 0 &&
                questions.map((q) => (
                  <div key={q.symptom} className="question-card">
                    <div className="question-text">{q.question}</div>
                    <div className="answer-btns">
                      <button
                        className="btn-yes"
                        onClick={() => handleAnswer(q.symptom, "yes")}
                        disabled={q.symptom in pendingAnswers}
                      >
                        ✓ Yes
                      </button>
                      <button
                        className="btn-no"
                        onClick={() => handleAnswer(q.symptom, "no")}
                        disabled={q.symptom in pendingAnswers}
                      >
                        ✗ No
                      </button>
                    </div>
                  </div>
                ))}

              {/* Submit answers */}
              {Object.keys(pendingAnswers).length > 0 && questions.length === 0 && (
                <button className="btn-primary" onClick={handleSubmitAnswers}>
                  Submit Answers & Continue
                </button>
              )}
            </div>

            {/* Right: Live hypotheses */}
            <div className="glass-card panel">
              <div className="panel-title">
                <span className="icon">🔬</span>
                Live Hypotheses
                <span className={`confidence-badge ${confidence}`} style={{ marginLeft: "auto" }}>
                  {confidence} confidence
                </span>
              </div>

              {diagnoses.map((d, i) => (
                <div key={d.disease} className="disease-item" style={{ animationDelay: `${i * 0.1}s` }}>
                  <div className="disease-header">
                    <span className="disease-name">
                      {i + 1}. {d.disease}
                    </span>
                    <span className="disease-score">{formatPct(d.confidence)}</span>
                  </div>
                  <div className="disease-bar-track">
                    <div
                      className="disease-bar-fill"
                      style={{ width: `${d.confidence * 100}%` }}
                    />
                  </div>
                  <div className="disease-symptoms">
                    {(d.matched_symptoms || []).slice(0, 4).map((s) => (
                      <span key={s} className="matched-sym">
                        ✓ {s.replace(/_/g, " ").toLowerCase()}
                      </span>
                    ))}
                    {(d.missing_symptoms || []).slice(0, 2).map((s) => (
                      <span key={s} className="missing-sym">
                        ? {s.replace(/_/g, " ").toLowerCase()}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── RESULT PHASE ────────────────────────────── */}
        {phase === "result" && (
          <div style={{ maxWidth: 800, margin: "0 auto" }}>
            <div className="glass-card final-result">
              <div className="result-icon">🩺</div>
              <h2>Diagnosis Complete</h2>
              {finalDiag && (
                <>
                  <div className="result-disease">{finalDiag.disease}</div>
                  <div className="result-score">
                    {formatPct(finalDiag.confidence || finalDiag.score || 0)}
                  </div>
                  <span className={`confidence-badge ${confidenceColor(finalDiag.confidence || finalDiag.score || 0)}`}>
                    {confidenceColor(finalDiag.confidence || finalDiag.score || 0)} confidence
                  </span>
                </>
              )}
            </div>

            {/* All ranked diagnoses */}
            <div className="result-details">
              <div className="glass-card result-section">
                <h3>🏥 All Considered Diagnoses</h3>
                {diagnoses.map((d, i) => (
                  <div key={d.disease} className="disease-item">
                    <div className="disease-header">
                      <span className="disease-name">
                        {i + 1}. {d.disease}
                      </span>
                      <span className="disease-score">{formatPct(d.confidence)}</span>
                    </div>
                    <div className="disease-bar-track">
                      <div className="disease-bar-fill" style={{ width: `${d.confidence * 100}%` }} />
                    </div>
                    <div className="disease-symptoms">
                      {(d.matched_symptoms || []).map((s) => (
                        <span key={s} className="matched-sym">
                          ✓ {s.replace(/_/g, " ").toLowerCase()}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Matched symptoms detail */}
              {finalDiag?.matched_symptoms && (
                <div className="glass-card result-section">
                  <h3>✅ Matched Symptoms</h3>
                  <div className="disease-symptoms" style={{ gap: "0.4rem" }}>
                    {finalDiag.matched_symptoms.map((s) => (
                      <span key={s} className="matched-sym" style={{ fontSize: "0.8rem", padding: "0.25rem 0.7rem" }}>
                        {s.replace(/_/g, " ").toLowerCase()}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Literature support */}
              {finalDiag?.literature_support && finalDiag.literature_support.length > 0 && (
                <div className="glass-card result-section">
                  <h3>📚 Literature Support (RAG)</h3>
                  {finalDiag.literature_support.map((lit, i) => (
                    <div key={i} style={{ marginBottom: "0.75rem" }}>
                      <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "var(--accent-light)" }}>
                        {lit.title}
                      </div>
                      <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                        {lit.snippet}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={{ textAlign: "center" }}>
              <button className="btn-secondary" onClick={handleReset}>
                🔄 Start New Diagnosis
              </button>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="disclaimer">
          ⚕️ This is an AI research tool for educational purposes only. It does not provide professional medical advice.
        </div>
      </div>
    </>
  );
}
