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
import { useLanguageContext } from "../contexts/LanguageContext";

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
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [exercise, setExercise] = useState<RewriteExerciseResponse | null>(null);
  const [stats, setStats] = useState<RewriteStatsResponse | null>(null);
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState<string | null>(null);
  const [skipMessage, setSkipMessage] = useState("");
  const [noMistakesMessage, setNoMistakesMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [seenMistakeIds, setSeenMistakeIds] = useState<number[]>([]);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const loadExercise = async (options?: {
    postMessage?: string;
    excludeIds?: number[];
  }) => {
    if (!currentLanguageProfile) return;

    setLoading(true);
    setFeedback(null);
    setSkipMessage("");
    setNoMistakesMessage(null);
    setAnswer("");
    setHasSubmitted(false);
    try {
      const ex = await getRewriteExercise(
        currentLanguageProfile.language_code,
        1,
        options?.excludeIds
      );
      setExercise(ex);
      if (ex && !seenMistakeIds.includes(ex.source_mistake_id)) {
        setSeenMistakeIds((prev) => [...prev, ex.source_mistake_id]);
      }
      if (options?.postMessage) {
        setSkipMessage(options.postMessage);
      }
      if (!ex && options?.postMessage) {
        setNoMistakesMessage("No mistakes to correct at this time!");
      }
    } catch (err: any) {
      setExercise(null);
      const message =
        err?.message && /no mistakes/i.test(err.message)
          ? "No mistakes to correct at this time!"
          : null;
      setNoMistakesMessage(message);
      if (message) {
        setSeenMistakeIds([]);
      }
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    if (!currentLanguageProfile) return;

    try {
      const st = await getRewriteStats(currentLanguageProfile.language_code);
      setStats(st);
    } catch {
      setStats(null);
    }
  };

  useEffect(() => {
    if (!currentLanguageProfile) return;

    loadExercise();
    loadStats();
  }, [currentLanguageProfile]);

  const handleSubmit = async () => {
    if (!exercise || !answer.trim() || !currentLanguageProfile) return;
    setLoading(true);
    try {
      const result = await submitRewriteExercise({
        user_id: 1,
        language_code: currentLanguageProfile.language_code,
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
      setHasSubmitted(true);
    } catch (err: any) {
      setFeedback(err?.message || "Could not submit rewrite.");
    } finally {
      setLoading(false);
    }
  };

  if (isLoadingLanguage) {
    return <p style={{ color: "#94a3b8", textAlign: "center" }}>Loading language profile...</p>;
  }

  if (!currentLanguageProfile) {
    return (
      <p style={{ color: "#94a3b8", textAlign: "center" }}>
        Please select or create a language profile to start rewriting exercises.
      </p>
    );
  }

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>
        Rewrite Practice ({currentLanguageProfile.display_name})
      </h1>
      <p style={{ color: "#64748b", marginBottom: 16 }}>
        You get your original incorrect sentence and rewrite it correctly.
      </p>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: 8 }}>Exercise</h3>
        {loading && !exercise ? (
          <p style={{ color: "#94a3b8" }}>Loading exercise...</p>
        ) : exercise ? (
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
              <button
                style={btnPrimary}
                onClick={
                  hasSubmitted
                    ? () =>
                        loadExercise({
                          postMessage: "Next sentence loaded.",
                          excludeIds: [...seenMistakeIds],
                        })
                    : handleSubmit
                }
                disabled={loading || (!hasSubmitted && !answer.trim())}
              >
                {hasSubmitted ? "Next Sentence" : "Submit Rewrite"}
              </button>
              {!hasSubmitted && (
                <button
                  style={{ ...btnPrimary, background: "#0ea5e9" }}
                  onClick={() =>
                    loadExercise({
                      postMessage: "Skipped to the next mistake.",
                      excludeIds: [...seenMistakeIds],
                    })
                  }
                  disabled={loading}
                >
                  Skip
                </button>
              )}
            </div>
            {(feedback || skipMessage) && (
              <p style={{ marginTop: 10, color: "#334155", fontSize: 14 }}>
                {feedback || skipMessage}
              </p>
            )}
          </>
        ) : (
          <p style={{ color: "#94a3b8" }}>
            {noMistakesMessage ?? "No rewrite exercises available yet. Analyze a few sessions first."}
          </p>
        )}
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: 8 }}>Rewrite Stats</h3>
        {loading && !stats ? (
          <p style={{ color: "#94a3b8" }}>Loading stats...</p>
        ) : !stats ? (
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
