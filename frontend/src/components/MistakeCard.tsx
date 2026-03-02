import type { MistakeOut } from "../types";

const cardStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  marginBottom: 12,
};

const badgeStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  background: "#eef2ff",
  color: "#4338ca",
  marginRight: 8,
};

const spanStyle: React.CSSProperties = {
  background: "#fee2e2",
  color: "#b91c1c",
  padding: "1px 4px",
  borderRadius: 3,
  fontWeight: 500,
};

const correctionStyle: React.CSSProperties = {
  background: "#dcfce7",
  color: "#166534",
  padding: "1px 4px",
  borderRadius: 3,
  fontWeight: 500,
};

interface Props {
  mistake: MistakeOut;
}

export default function MistakeCard({ mistake }: Props) {
  return (
    <div style={cardStyle}>
      <div style={{ marginBottom: 8 }}>
        <span style={badgeStyle}>{mistake.mistake_type.label}</span>
        {mistake.confidence != null && (
          <span style={{ fontSize: 12, color: "#64748b" }}>
            Confidence: {(mistake.confidence * 100).toFixed(0)}%
          </span>
        )}
        {mistake.stt_uncertain && (
          <span
            style={{
              ...badgeStyle,
              background: "#fef3c7",
              color: "#92400e",
              marginLeft: 4,
            }}
          >
            STT uncertain
          </span>
        )}
        {mistake.uncertain && (
          <span
            style={{
              ...badgeStyle,
              background: "#fef3c7",
              color: "#92400e",
              marginLeft: 4,
            }}
          >
            Uncertain
          </span>
        )}
      </div>

      <div style={{ marginBottom: 6 }}>
        <span style={spanStyle}>{mistake.transcript_span}</span>
        {mistake.suggested_correction && (
          <>
            {" → "}
            <span style={correctionStyle}>{mistake.suggested_correction}</span>
          </>
        )}
      </div>

      {mistake.explanation_short && (
        <p style={{ fontSize: 14, color: "#475569", margin: 0 }}>
          {mistake.explanation_short}
        </p>
      )}

      {mistake.uncertain_reason && (
        <p style={{ fontSize: 13, color: "#92400e", marginTop: 4 }}>
          Note: {mistake.uncertain_reason}
        </p>
      )}
    </div>
  );
}
