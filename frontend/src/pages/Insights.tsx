import { useEffect, useMemo, useState } from "react";
import { getInsights } from "../services/api";
import TrendChart from "../components/TrendChart";
import type { InsightsResponse, SpeakingWinItem } from "../types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";
import { useLanguageContext } from "../contexts/LanguageContext";

const cardStyle: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 16,
  textAlign: "center",
  minWidth: 120,
};

const speakingWinDetailStyle: React.CSSProperties = {
  marginTop: 12,
  background: "#fff",
  border: "1px solid #d1fae5",
  borderRadius: 8,
  padding: 16,
};

function SpeakingWinDetails({ win }: { win: SpeakingWinItem }) {
  return (
    <div style={speakingWinDetailStyle}>
      <div style={{ fontSize: 13, color: "#475569", marginBottom: 8 }}>
        Focus area: <strong>{win.focus_label}</strong>
      </div>
      <div style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>{win.created_at}</div>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "#991b1b", marginBottom: 4 }}>
          Earlier problematic transcript
        </div>
        <div style={{ color: "#334155", lineHeight: 1.7 }}>
          {win.previous_bad_sentence || "No earlier transcript captured."}
        </div>
      </div>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "#166534", marginBottom: 4 }}>
          Improved transcript
        </div>
        <div style={{ color: "#334155", lineHeight: 1.7 }}>
          {win.improved_sentence || "No improved transcript captured."}
        </div>
      </div>
      {win.reason ? (
        <div style={{ fontSize: 13, color: "#475569", marginBottom: 12 }}>{win.reason}</div>
      ) : null}
      {win.previous_wrong_span || win.suggested_correction ? (
        <div style={{ fontSize: 13, color: "#475569" }}>
          {win.previous_wrong_span ? (
            <span>
              Earlier issue: <strong>{win.previous_wrong_span}</strong>
            </span>
          ) : null}
          {win.previous_wrong_span && win.suggested_correction ? " -> " : ""}
          {win.suggested_correction ? (
            <span>
              Suggested fix: <strong>{win.suggested_correction}</strong>
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export default function Insights() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredCode, setHoveredCode] = useState<string | null>(null);
  const [selectedSpeakingWinId, setSelectedSpeakingWinId] = useState<number | null>(null);

  useEffect(() => {
    if (!currentLanguageProfile) {
      if (!isLoadingLanguage) setLoading(false);
      return;
    }

    setLoading(true);
    getInsights(10, 30, currentLanguageProfile.id)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [currentLanguageProfile, isLoadingLanguage]);

  useEffect(() => {
    if (!data?.latest_speaking_win) {
      setSelectedSpeakingWinId(null);
      return;
    }
    setSelectedSpeakingWinId((prev) => prev ?? null);
  }, [data]);

  const previousSpeakingWins = useMemo(() => {
    if (!data) {
      return [];
    }
    if (!data.latest_speaking_win) {
      return data.speaking_win_history;
    }
    return data.speaking_win_history.filter(
      (item) => item.event_id !== data.latest_speaking_win?.event_id
    );
  }, [data]);

  if (loading) {
    return <p style={{ color: "#94a3b8", textAlign: "center" }}>Loading...</p>;
  }

  if (!currentLanguageProfile) {
    return (
      <p style={{ color: "#94a3b8", textAlign: "center" }}>
        Please select or create a language profile to view insights.
      </p>
    );
  }
  if (!data) {
    return (
      <p style={{ color: "#94a3b8", textAlign: "center" }}>
        Could not load insights.
      </p>
    );
  }

  return (
    <div>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>
        Insights
      </h1>

      {data.latest_speaking_win && (
        <div style={{ marginBottom: 18 }}>
          <button
            type="button"
            onClick={() =>
              setSelectedSpeakingWinId((prev) =>
                prev === data.latest_speaking_win?.event_id ? null : data.latest_speaking_win?.event_id ?? null
              )
            }
            style={{
              width: "100%",
              textAlign: "left",
              background: "#ecfdf5",
              border: "1px solid #86efac",
              color: "#166534",
              borderRadius: 8,
              padding: "10px 12px",
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {data.latest_speaking_win.summary}
          </button>
          {selectedSpeakingWinId === data.latest_speaking_win.event_id ? (
            <SpeakingWinDetails win={data.latest_speaking_win} />
          ) : null}
        </div>
      )}

      {previousSpeakingWins.length > 0 && (
        <details style={{ marginBottom: 24 }}>
          <summary style={{ cursor: "pointer", fontWeight: 600, color: "#334155" }}>
            Previous speaking wins ({previousSpeakingWins.length})
          </summary>
          <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
            {previousSpeakingWins.map((win) => (
              <div key={win.event_id}>
                <button
                  type="button"
                  onClick={() =>
                    setSelectedSpeakingWinId((prev) => (prev === win.event_id ? null : win.event_id))
                  }
                  style={{
                    width: "100%",
                    textAlign: "left",
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                    borderRadius: 8,
                    padding: "10px 12px",
                    cursor: "pointer",
                    color: "#334155",
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{win.summary}</div>
                  <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>{win.created_at}</div>
                </button>
                {selectedSpeakingWinId === win.event_id ? <SpeakingWinDetails win={win} /> : null}
              </div>
            ))}
          </div>
        </details>
      )}

      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
        Top Mistake Types
      </h2>
      {data.top_mistakes.length === 0 ? (
        <p style={{ color: "#94a3b8", marginBottom: 24 }}>
          No mistakes recorded yet. Sessions with zero mistakes still appear in progress and history, but they do not create rewrite exercises.
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

      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
        Error Trends Over Sessions
      </h2>
      <div style={{ marginBottom: 24 }}>
        <TrendChart trends={data.trends} />
      </div>

      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
        Most Common Errors (Bar Chart)
      </h2>
      <div
        style={{
          marginBottom: 24,
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 8,
          padding: 16,
        }}
      >
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.top_mistakes}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="label" fontSize={12} />
            <YAxis allowDecimals={false} fontSize={12} />
            <Tooltip />
            <Bar dataKey="count" fill="#4338ca" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
        Improvement Over Time (Error Rate)
      </h2>
      <div
        style={{
          marginBottom: 24,
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 8,
          padding: 16,
        }}
      >
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data.progress}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" fontSize={12} />
            <YAxis allowDecimals fontSize={12} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="error_rate_per_100_words"
              stroke="#dc2626"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
