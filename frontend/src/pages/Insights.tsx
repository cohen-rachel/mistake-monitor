import { useEffect, useMemo, useState } from "react";
import { getInsights } from "../services/api";
import TrendChart, {
  type TrendRange,
  cutoffForRange,
  formatAxisDate,
  formatHoverDate,
  parseTrendDate,
  startOfTrendWeek,
} from "../components/TrendChart";
import type { InsightsResponse, SpeakingWinItem, MistakeCountItem } from "../types";
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

const rangeOptions: Array<{ value: TrendRange; label: string }> = [
  { value: "7d", label: "7 days" },
  { value: "4w", label: "4 weeks" },
  { value: "1y", label: "1 year" },
  { value: "all", label: "All time" },
];

function parseInsightsDate(value: string): Date {
  return parseTrendDate(value);
}

function humanizeCode(code: string): string {
  return code
    .replace(/-/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function filterByRange<T>(
  items: T[],
  range: TrendRange,
  getDate: (item: T) => string
): T[] {
  const cutoff = cutoffForRange(range);
  if (!cutoff) {
    return items;
  }
  return items.filter((item) => parseInsightsDate(getDate(item)) >= cutoff);
}

function RangeSelect({
  value,
  onChange,
}: {
  value: TrendRange;
  onChange: (value: TrendRange) => void;
}) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value as TrendRange)}
      style={{
        padding: "8px 10px",
        borderRadius: 8,
        border: "1px solid #cbd5e1",
        background: "#fff",
        color: "#334155",
        fontSize: 13,
      }}
    >
      {rangeOptions.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

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
  const [topRange, setTopRange] = useState<TrendRange>("all");
  const [barRange, setBarRange] = useState<TrendRange>("all");
  const [progressRange, setProgressRange] = useState<TrendRange>("all");
  const [trendRange, setTrendRange] = useState<TrendRange>("all");

  useEffect(() => {
    if (!currentLanguageProfile) {
      if (!isLoadingLanguage) setLoading(false);
      return;
    }

    setLoading(true);
    getInsights(10, 100, currentLanguageProfile.id)
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

  const allSpeakingWins = useMemo(() => data?.speaking_win_history ?? [], [data]);

  const metadataByCode = useMemo(() => {
    const map = new Map<string, MistakeCountItem>();
    data?.top_mistakes.forEach((item) => map.set(item.code, item));
    return map;
  }, [data]);

  const buildTopMistakesForRange = useMemo(() => {
    return (range: TrendRange): MistakeCountItem[] => {
      if (!data) {
        return [];
      }
      const filteredTrends = filterByRange(data.trends, range, (item) => item.date);
      const filteredRecentMistakes = filterByRange(
        data.recent_mistakes,
        range,
        (item) => item.date
      );

      const counts = new Map<string, number>();
      filteredTrends.forEach((item) => {
        counts.set(item.mistake_type_code, (counts.get(item.mistake_type_code) || 0) + item.count);
      });

      const recentSentenceByCode = new Map<string, string>();
      filteredRecentMistakes.forEach((item) => {
        if (!recentSentenceByCode.has(item.mistake_type_code)) {
          recentSentenceByCode.set(
            item.mistake_type_code,
            item.original_sentence || item.transcript_span || "No recent example yet."
          );
        }
      });

      return Array.from(counts.entries())
        .map(([code, count]) => {
          const meta = metadataByCode.get(code);
          return {
            code,
            label: meta?.label || humanizeCode(code),
            count,
            description: meta?.description,
            recent_mistake_summary:
              recentSentenceByCode.get(code) || meta?.recent_mistake_summary || "No recent example yet.",
          };
        })
        .sort((a, b) => b.count - a.count);
    };
  }, [data, metadataByCode]);

  const topMistakesForCards = useMemo(() => buildTopMistakesForRange(topRange), [buildTopMistakesForRange, topRange]);
  const topMistakesForBar = useMemo(() => buildTopMistakesForRange(barRange), [buildTopMistakesForRange, barRange]);
  const filteredProgress = useMemo(
    () => (data ? filterByRange(data.progress, progressRange, (item) => item.date) : []),
    [data, progressRange]
  );
  const progressChartData = useMemo(() => {
    const useWeeklyBuckets = progressRange === "1y" || progressRange === "all";
    const grouped = new Map<
      string,
      {
        bucketKey: string;
        bucketDate: Date;
        hoverLabel: string;
        totalErrorRate: number;
        pointCount: number;
      }
    >();

    for (const point of filteredProgress) {
      const pointDate = parseInsightsDate(point.date);
      const bucketDate = useWeeklyBuckets
        ? startOfTrendWeek(pointDate)
        : new Date(pointDate.getFullYear(), pointDate.getMonth(), pointDate.getDate());
      const bucketKey = bucketDate.toISOString().slice(0, 10);
      const existing = grouped.get(bucketKey);
      if (existing) {
        existing.totalErrorRate += point.error_rate_per_100_words;
        existing.pointCount += 1;
      } else {
        grouped.set(bucketKey, {
          bucketKey,
          bucketDate,
          hoverLabel: formatHoverDate(bucketDate),
          totalErrorRate: point.error_rate_per_100_words,
          pointCount: 1,
        });
      }
    }

    return Array.from(grouped.values())
      .sort((a, b) => a.bucketDate.getTime() - b.bucketDate.getTime())
      .map((item) => ({
        bucketKey: item.bucketKey,
        hoverLabel: item.hoverLabel,
        error_rate_per_100_words: Number((item.totalErrorRate / item.pointCount).toFixed(2)),
      }));
  }, [filteredProgress, progressRange]);
  const progressFirstTick = progressChartData[0]?.bucketKey;
  const progressLastTick = progressChartData[progressChartData.length - 1]?.bucketKey;

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

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>
          Top Mistake Types
        </h2>
        <RangeSelect value={topRange} onChange={setTopRange} />
      </div>
      {topMistakesForCards.length === 0 ? (
        <p style={{ color: "#94a3b8", marginBottom: 24 }}>
          No mistakes recorded yet in this time range.
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
          {topMistakesForCards.map((item) => (
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

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>
          Error Trends Over Time
        </h2>
        <RangeSelect value={trendRange} onChange={setTrendRange} />
      </div>
      <div style={{ marginBottom: 24 }}>
        <TrendChart trends={data.trends} range={trendRange} />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>
          Most Common Errors
        </h2>
        <RangeSelect value={barRange} onChange={setBarRange} />
      </div>
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
          <BarChart data={topMistakesForBar}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="label" fontSize={12} />
            <YAxis allowDecimals={false} fontSize={12} />
            <Tooltip />
            <Bar dataKey="count" fill="#4338ca" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>
          Errors per 100 words
        </h2>
        <RangeSelect value={progressRange} onChange={setProgressRange} />
      </div>
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
          <LineChart data={progressChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="bucketKey"
              fontSize={12}
              ticks={
                progressFirstTick === progressLastTick
                  ? [progressFirstTick].filter(Boolean)
                  : [progressFirstTick, progressLastTick].filter(Boolean)
              }
              tickFormatter={(value) => formatAxisDate(new Date(`${value}T00:00:00`))}
            />
            <YAxis allowDecimals fontSize={12} />
            <Tooltip
              formatter={(value: number) => [value, "Error Rate"]}
              labelFormatter={(_, payload) => {
                const row = payload?.[0]?.payload as { hoverLabel?: string } | undefined;
                return row?.hoverLabel || "";
              }}
            />
            <Line
              type="monotone"
              dataKey="error_rate_per_100_words"
              name="Error Rate"
              stroke="#dc2626"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <details style={{ marginBottom: 24 }}>
        <summary style={{ cursor: "pointer", fontWeight: 600, color: "#334155" }}>
          Common Error Patterns ({data.common_patterns.length})
        </summary>
        <div style={{ marginTop: 12 }}>
          {data.common_patterns.length === 0 ? (
            <p style={{ color: "#94a3b8", margin: 0 }}>
              No repeated construction-level patterns found yet.
            </p>
          ) : (
            <div style={{ display: "grid", gap: 10 }}>
              {data.common_patterns.map((item) => (
                <div
                  key={item.code}
                  style={{
                    background: "#fff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 8,
                    padding: "12px 14px",
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 16,
                    alignItems: "flex-start",
                  }}
                >
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: "#1e293b", marginBottom: 4 }}>
                      {item.label}
                    </div>
                    {item.description ? (
                      <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6 }}>
                        Broader type: {item.description}
                      </div>
                    ) : null}
                    <div style={{ fontSize: 13, color: "#475569", lineHeight: 1.6 }}>
                      {item.recent_mistake_summary || "No recent example yet."}
                    </div>
                  </div>
                  <div
                    style={{
                      minWidth: 44,
                      textAlign: "center",
                      fontSize: 24,
                      fontWeight: 700,
                      color: "#4338ca",
                    }}
                  >
                    {item.count}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </details>

      <div>
      {allSpeakingWins.length > 0 && (
        <details style={{ marginBottom: 24 }}>
          <summary style={{ cursor: "pointer", fontWeight: 600, color: "#334155" }}>
            Recent Improvement Wins ({allSpeakingWins.length})
          </summary>
          <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
            {allSpeakingWins.map((win) => (
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
      </div>
    </div>
  );
}
