import { useEffect, useState } from "react";
import { getInsights } from "../services/api";
import TrendChart from "../components/TrendChart";
import type { InsightsResponse } from "../types";

const cardStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  textAlign: "center",
  minWidth: 120,
};

const correctionRow: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 12,
  marginBottom: 8,
  cursor: "pointer",
};

const spanErr: React.CSSProperties = {
  background: "#fee2e2",
  color: "#b91c1c",
  padding: "1px 4px",
  borderRadius: 3,
  fontWeight: 500,
};

const spanFix: React.CSSProperties = {
  background: "#dcfce7",
  color: "#166534",
  padding: "1px 4px",
  borderRadius: 3,
  fontWeight: 500,
};

export default function Insights() {
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredCode, setHoveredCode] = useState<string | null>(null);
  const [selectedMistakeId, setSelectedMistakeId] = useState<number | null>(null);

  useEffect(() => {
    getInsights(10, 30)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p style={{ color: "#94a3b8", textAlign: "center" }}>Loading...</p>;
  }

  if (!data) {
    return (
      <p style={{ color: "#94a3b8", textAlign: "center" }}>
        Could not load insights.
      </p>
    );
  }

  const selectedMistake =
    data.recent_mistakes.find((m) => m.id === selectedMistakeId) ?? null;

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>
        Insights
      </h1>

      {/* Top Mistakes */}
      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
        Top Mistake Types
      </h2>
      {data.top_mistakes.length === 0 ? (
        <p style={{ color: "#94a3b8", marginBottom: 24 }}>
          No mistakes recorded yet. Analyze some sessions to see insights.
        </p>
      ) : (
        <div
          style={{
            display: "flex",
            gap: 12,
            flexWrap: "wrap",
            marginBottom: 24,
          }}
        >
          {data.top_mistakes.map((item) => (
            <div
              key={item.code}
              style={{ ...cardStyle, position: "relative" }}
              onMouseEnter={() => setHoveredCode(item.code)}
              onMouseLeave={() => setHoveredCode(null)}
            >
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: "#4338ca",
                }}
              >
                {item.count}
              </div>
              <div style={{ fontSize: 13, color: "#475569", fontWeight: 600 }}>
                {item.label}
              </div>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>{item.code}</div>
              {hoveredCode === item.code && (
                <div
                  style={{
                    position: "absolute",
                    top: "100%",
                    left: 0,
                    zIndex: 20,
                    width: 280,
                    marginTop: 8,
                    textAlign: "left",
                    background: "#0f172a",
                    color: "#e2e8f0",
                    borderRadius: 8,
                    padding: 10,
                    boxShadow: "0 8px 24px rgba(0,0,0,0.25)",
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 6 }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: 12, marginBottom: 8 }}>
                    {item.description || "No category description yet."}
                  </div>
                  <div style={{ fontSize: 11, color: "#94a3b8" }}>
                    Most recent example:
                  </div>
                  <div style={{ fontSize: 12 }}>
                    {item.recent_mistake_summary || "No recent example yet."}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Trend Chart */}
      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
        Error Trends Over Sessions
      </h2>
      <div style={{ marginBottom: 24 }}>
        <TrendChart trends={data.trends} />
      </div>

      {/* Recent Mistakes */}
      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
        Recent Mistakes & Corrections
      </h2>
      {data.recent_mistakes.length === 0 ? (
        <p style={{ color: "#94a3b8" }}>No recent mistakes.</p>
      ) : (
        data.recent_mistakes.map((item) => (
          <div
            key={item.id}
            style={{
              ...correctionRow,
              border:
                selectedMistakeId === item.id
                  ? "2px solid #4338ca"
                  : correctionRow.border,
            }}
            onClick={() =>
              setSelectedMistakeId((prev) => (prev === item.id ? null : item.id))
            }
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 6,
              }}
            >
              <span
                style={{
                  fontSize: 12,
                  color: "#4338ca",
                  fontWeight: 600,
                }}
              >
                {item.mistake_type_label}
              </span>
              <span style={{ fontSize: 12, color: "#94a3b8" }}>
                {item.date} &middot; Session #{item.session_id}
              </span>
            </div>
            <div style={{ marginBottom: 4 }}>
              {item.transcript_span && (
                <span style={spanErr}>{item.transcript_span}</span>
              )}
              {item.suggested_correction && (
                <>
                  {" → "}
                  <span style={spanFix}>{item.suggested_correction}</span>
                </>
              )}
            </div>
            {item.explanation_short && (
              <p style={{ fontSize: 13, color: "#475569", margin: 0 }}>
                {item.explanation_short}
              </p>
            )}
          </div>
        ))
      )}

      {selectedMistake && (
        <div
          style={{
            marginTop: 16,
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
            padding: 16,
          }}
        >
          <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>
            Mistake Detail
          </h3>
          <div style={{ marginBottom: 8, fontSize: 13, color: "#64748b" }}>
            Original sentence
          </div>
          <div style={{ marginBottom: 12, fontSize: 15, lineHeight: 1.7 }}>
            {selectedMistake.original_sentence || "No sentence context found."}
          </div>

          <div style={{ marginBottom: 8, fontSize: 13, color: "#64748b" }}>
            Corrected sentence
          </div>
          <div style={{ marginBottom: 12, fontSize: 15, lineHeight: 1.7 }}>
            {selectedMistake.corrected_sentence || "No corrected sentence available."}
          </div>

          {selectedMistake.transcript_span && selectedMistake.suggested_correction && (
            <div style={{ fontSize: 14 }}>
              <span
                style={{
                  textDecoration: "line-through",
                  color: "#b91c1c",
                  background: "#fee2e2",
                  padding: "2px 4px",
                  borderRadius: 4,
                  marginRight: 6,
                }}
              >
                {selectedMistake.transcript_span}
              </span>
              <span style={{ color: "#64748b", marginRight: 6 }}>→</span>
              <span
                style={{
                  color: "#166534",
                  background: "#dcfce7",
                  padding: "2px 4px",
                  borderRadius: 4,
                  fontWeight: 600,
                }}
              >
                {selectedMistake.suggested_correction}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
