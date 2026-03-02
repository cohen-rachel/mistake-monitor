import { useEffect, useState } from "react";
import {
  getRewriteExercise,
  submitRewriteExercise,
  getRewriteStats,
} from "../services/api";
import type {
  RewriteExerciseResponse,
  RewriteStatsResponse,
} from "../types";

const cardStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 10,
  padding: 16,
  marginBottom: 16,
};

const btnPrimary: React.CSSProperties = {
  padding: "10px 16px",
  borderRadius: 8,
  border: "none",
  background: "#4338ca",
  color: "#fff",
  fontWeight: 600,
  cursor: "pointer",
};

export default function Rewrite() {
  const [language, setLanguage] = useState("en");
  const [exercise, setExercise] = useState<RewriteExerciseResponse | null>(null);
  const [stats, setStats] = useState<RewriteStatsResponse | null>(null);
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadExercise = async () => {
    setLoading(true);
    setFeedback(null);
    setAnswer("");
    try {
      const ex = await getRewriteExercise(language);
      setExercise(ex);
    } catch {
      setExercise(null);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const st = await getRewriteStats(language);
      setStats(st);
    } catch {
      setStats(null);
    }
  };

  useEffect(() => {
    loadExercise();
    loadStats();
  }, [language]);

  const handleSubmit = async () => {
    if (!exercise || !answer.trim()) return;
    setLoading(true);
    try {
      const result = await submitRewriteExercise({
        user_id: 1,
        language,
        source_mistake_id: exercise.source_mistake_id,
        original_sentence: exercise.original_sentence,
        wrong_span: exercise.wrong_span,
        expected_correction: exercise.expected_correction,
        user_rewrite: answer.trim(),
      });
      const base = result.is_correct ? "Correct." : "Not quite.";
      const expected = result.expected_correction
        ? ` Expected correction: ${result.expected_correction}`
        : "";
      setFeedback(
        `${base} Score ${Math.round(result.score * 100)}%. ${result.feedback}${expected}`
      );
      await loadStats();
    } catch (err: any) {
      setFeedback(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>
        Rewrite Practice
      </h1>
      <p style={{ color: "#64748b", marginBottom: 16 }}>
        You get your original incorrect sentence and rewrite it correctly.
      </p>

      <div style={cardStyle}>
        <label style={{ fontSize: 14, color: "#475569", marginRight: 8 }}>
          Language:
        </label>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid #cbd5e1" }}
        >
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="ja">Japanese</option>
          <option value="de">German</option>
          <option value="it">Italian</option>
          <option value="pt">Portuguese</option>
        </select>
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: 8 }}>Exercise</h3>
        {exercise ? (
          <>
            <div style={{ fontSize: 13, color: "#64748b", marginBottom: 8 }}>
              {exercise.mistake_type_label}
            </div>
            <div style={{ fontSize: 16, lineHeight: 1.7, marginBottom: 10 }}>
              {exercise.original_sentence}
            </div>
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="Rewrite the sentence correctly..."
              style={{
                width: "100%",
                minHeight: 90,
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                padding: 10,
                fontSize: 14,
                marginBottom: 10,
              }}
            />
            <div style={{ display: "flex", gap: 8 }}>
              <button style={btnPrimary} onClick={handleSubmit} disabled={loading || !answer.trim()}>
                Submit Rewrite
              </button>
              <button
                style={{ ...btnPrimary, background: "#0ea5e9" }}
                onClick={loadExercise}
                disabled={loading}
              >
                Next Sentence
              </button>
            </div>
            {feedback && (
              <p style={{ marginTop: 10, color: "#334155", fontSize: 14 }}>{feedback}</p>
            )}
          </>
        ) : (
          <p style={{ color: "#94a3b8" }}>
            No rewrite exercises available yet. Analyze a few sessions first.
          </p>
        )}
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: 8 }}>Rewrite Stats</h3>
        {!stats ? (
          <p style={{ color: "#94a3b8" }}>No stats yet.</p>
        ) : (
          <>
            <p style={{ marginBottom: 8, color: "#334155" }}>
              Overall accuracy: <strong>{Math.round(stats.overall_accuracy * 100)}%</strong> (
              {stats.total_correct}/{stats.total_attempts})
            </p>
            {stats.recent_attempts.slice(0, 8).map((item, idx) => (
              <div
                key={idx}
                style={{
                  borderTop: idx === 0 ? "none" : "1px solid #e2e8f0",
                  paddingTop: idx === 0 ? 0 : 8,
                  marginTop: idx === 0 ? 0 : 8,
                  fontSize: 13,
                  color: "#475569",
                }}
              >
                <span style={{ color: "#b91c1c" }}>{item.wrong_span || "(unknown)"}</span>
                {" -> "}
                <span style={{ color: "#166534" }}>{item.expected_correction || "(unknown)"}</span>
                {" · accuracy "}
                <strong>{Math.round(item.accuracy * 100)}%</strong>
                {item.latest_result !== undefined && (
                  <span>{item.latest_result ? " · latest correct" : " · latest incorrect"}</span>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
