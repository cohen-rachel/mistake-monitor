import React, { memo, useEffect, useMemo, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import Screen from "../components/Screen";
import SectionCard from "../components/SectionCard";
import VerticalBarChart from "../components/VerticalBarChart";
import PieChart from "../components/PieChart";
import TimeSeriesChart from "../components/TimeSeriesChart";
import StackedTrendChart from "../components/StackedTrendChart";
import SelectField from "../components/SelectField";
import { getInsights } from "../services/api";
import { useLanguageContext } from "../contexts/LanguageContext";
import { useLandingState } from "../contexts/LandingStateContext";
import type { InsightsResponse, MistakeCountItem, SpeakingWinItem } from "../types";
import { colors } from "../theme";

type TrendRange = "7d" | "4w" | "1y" | "all";

const rangeOptions = [
  { label: "7 days", value: "7d" },
  { label: "4 weeks", value: "4w" },
  { label: "1 year", value: "1y" },
  { label: "All time", value: "all" },
];

function parseInsightsDate(value: string): Date {
  return new Date(value.replace(" ", "T"));
}

function cutoffForRange(range: TrendRange): Date | null {
  const now = new Date();
  const cutoff = new Date(now);
  if (range === "7d") {
    cutoff.setDate(now.getDate() - 7);
    return cutoff;
  }
  if (range === "4w") {
    cutoff.setDate(now.getDate() - 28);
    return cutoff;
  }
  if (range === "1y") {
    cutoff.setFullYear(now.getFullYear() - 1);
    return cutoff;
  }
  return null;
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

function startOfWeek(date: Date): Date {
  const next = new Date(date);
  const day = next.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  next.setDate(next.getDate() + diff);
  next.setHours(0, 0, 0, 0);
  return next;
}

function formatShortDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(date);
}

function humanizeCode(code: string): string {
  return code
    .replace(/-/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function SpeakingWinDetails({ win }: { win: SpeakingWinItem }) {
  return (
    <View style={styles.detailCard}>
      <Text style={styles.metaLabel}>
        Focus area: <Text style={styles.metaValue}>{win.focus_label}</Text>
      </Text>
      <Text style={styles.metaDate}>{win.created_at}</Text>
      <Text style={styles.detailHeadingBad}>Earlier problematic transcript</Text>
      <Text style={styles.detailText}>
        {win.previous_bad_sentence || "No earlier transcript captured."}
      </Text>
      <Text style={styles.detailHeadingGood}>Improved transcript</Text>
      <Text style={styles.detailText}>
        {win.improved_sentence || "No improved transcript captured."}
      </Text>
      {win.reason ? <Text style={styles.recentMeta}>{win.reason}</Text> : null}
      {win.previous_wrong_span || win.suggested_correction ? (
        <Text style={styles.metaLabel}>
          Earlier issue: <Text style={styles.metaValue}>{win.previous_wrong_span || "Unknown"}</Text>
          {win.suggested_correction ? ` -> Suggested fix: ${win.suggested_correction}` : ""}
        </Text>
      ) : null}
    </View>
  );
}

function InsightsScreen() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const { dataRefreshVersion } = useLandingState();
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [winsExpanded, setWinsExpanded] = useState(false);
  const [patternsExpanded, setPatternsExpanded] = useState(false);
  const [selectedSpeakingWinId, setSelectedSpeakingWinId] = useState<number | null>(null);
  const [topRange, setTopRange] = useState<TrendRange>("all");
  const [trendRange, setTrendRange] = useState<TrendRange>("all");
  const [progressRange, setProgressRange] = useState<TrendRange>("all");

  useEffect(() => {
    if (!currentLanguageProfile) {
      if (!isLoadingLanguage) {
        setLoading(false);
      }
      return;
    }
    setLoading(true);
    getInsights(10, 100, currentLanguageProfile.id)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [currentLanguageProfile, isLoadingLanguage, dataRefreshVersion]);

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
              recentSentenceByCode.get(code) ||
              meta?.recent_mistake_summary ||
              "No recent example yet.",
          };
        })
        .sort((a, b) => b.count - a.count);
    };
  }, [data, metadataByCode]);

  const topMistakesForRange = useMemo(
    () => buildTopMistakesForRange(topRange),
    [buildTopMistakesForRange, topRange]
  );

  const topMistakeItems = useMemo(
    () =>
      topMistakesForRange.map((item, index) => ({
        key: `top-${item.code}-${index}`,
        label: item.label,
        value: item.count,
        accent: ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"][index % 5],
        helper: item.recent_mistake_summary || item.code,
      })),
    [topMistakesForRange]
  );

  const topMistakePie = useMemo(
    () =>
      topMistakesForRange.slice(0, 5).map((item, index) => ({
        key: `pie-${item.code}-${index}`,
        label: item.label,
        value: item.count,
        color: ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"][index % 5],
      })),
    [topMistakesForRange]
  );

  const trendItems = useMemo(() => {
    return data?.trends ?? [];
  }, [data]);

  const progressItems = useMemo(() => {
    if (!data) {
      return [];
    }
    const filtered = filterByRange(data.progress, progressRange, (item) => item.date);
    const useWeeklyBuckets = progressRange === "1y" || progressRange === "all";
    const grouped = new Map<
      string,
      {
        bucketDate: Date;
        totalErrorRate: number;
        pointCount: number;
        totalMistakes: number;
        totalWords: number;
      }
    >();

    filtered.forEach((point) => {
      const date = parseInsightsDate(point.date);
      const bucketDate = useWeeklyBuckets
        ? startOfWeek(date)
        : new Date(date.getFullYear(), date.getMonth(), date.getDate());
      const key = bucketDate.toISOString().slice(0, 10);
      const existing = grouped.get(key);
      if (existing) {
        existing.totalErrorRate += point.error_rate_per_100_words;
        existing.pointCount += 1;
        existing.totalMistakes += point.mistake_count;
        existing.totalWords += point.word_count;
      } else {
        grouped.set(key, {
          bucketDate,
          totalErrorRate: point.error_rate_per_100_words,
          pointCount: 1,
          totalMistakes: point.mistake_count,
          totalWords: point.word_count,
        });
      }
    });

    return Array.from(grouped.entries())
      .sort((a, b) => a[1].bucketDate.getTime() - b[1].bucketDate.getTime())
      .map(([key, value]) => ({
        key: `progress-${key}`,
        label: formatShortDate(value.bucketDate),
        value: Number((value.totalErrorRate / value.pointCount).toFixed(2)),
        helper: `${value.totalMistakes} mistakes across ${value.totalWords} words`,
      }));
  }, [data, progressRange]);

  return (
    <Screen>
      <Text style={styles.title}>Insights</Text>
      {loading ? (
        <SectionCard>
          <Text style={styles.muted}>Loading...</Text>
        </SectionCard>
      ) : !currentLanguageProfile ? (
        <SectionCard>
          <Text style={styles.muted}>
            Please select or create a language profile to view insights.
          </Text>
        </SectionCard>
      ) : !data ? (
        <SectionCard>
          <Text style={styles.muted}>Could not load insights.</Text>
        </SectionCard>
      ) : (
        <>
          {data.latest_speaking_win ? (
            <View style={styles.bannerList}>
              <Pressable
                onPress={() =>
                  setSelectedSpeakingWinId((prev) =>
                    prev === data.latest_speaking_win?.event_id
                      ? null
                      : data.latest_speaking_win?.event_id ?? null
                  )
                }
                style={styles.banner}
              >
                <Text style={styles.bannerText}>{data.latest_speaking_win.summary}</Text>
              </Pressable>
              {selectedSpeakingWinId === data.latest_speaking_win.event_id ? (
                <SpeakingWinDetails win={data.latest_speaking_win} />
              ) : null}
            </View>
          ) : null}

          <SectionCard>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Top Mistake Types</Text>
              <SelectField
                value={topRange}
                options={rangeOptions}
                onChange={(value) => setTopRange(value as TrendRange)}
                compact
              />
            </View>
            <VerticalBarChart
              data={topMistakeItems}
              emptyMessage="No mistakes recorded yet in this time range."
            />
          </SectionCard>

          <SectionCard>
            <Text style={styles.sectionTitle}>Mistake Distribution</Text>
            <PieChart
              data={topMistakePie}
              emptyMessage="No mistake distribution yet."
            />
          </SectionCard>

          <SectionCard>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Errors per 100 words</Text>
              <SelectField
                value={progressRange}
                options={rangeOptions}
                onChange={(value) => setProgressRange(value as TrendRange)}
                compact
              />
            </View>
            <TimeSeriesChart
              data={progressItems}
              emptyMessage="No progress data yet."
            />
          </SectionCard>

          <SectionCard>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Error Trends Over Time</Text>
              <SelectField
                value={trendRange}
                options={rangeOptions}
                onChange={(value) => setTrendRange(value as TrendRange)}
                compact
              />
            </View>
            <StackedTrendChart
              trends={trendItems}
              range={trendRange}
              emptyMessage="No trend data yet. Analyze some sessions to see trends."
            />
          </SectionCard>

          <SectionCard>
            <Pressable
              onPress={() => setPatternsExpanded((prev) => !prev)}
              style={styles.expandHeader}
            >
              <Text style={styles.sectionTitle}>
                Common Error Patterns ({data.common_patterns.length})
              </Text>
              <Text style={styles.expandToggle}>{patternsExpanded ? "Hide" : "Show"}</Text>
            </Pressable>
            {patternsExpanded ? (
              data.common_patterns.length === 0 ? (
                <Text style={styles.muted}>
                  No repeated construction-level patterns found yet.
                </Text>
              ) : (
                data.common_patterns.map((item) => (
                  <View key={item.code} style={styles.patternRow}>
                    <View style={styles.patternCopy}>
                      <Text style={styles.patternTitle}>{item.label}</Text>
                      {item.description ? (
                        <Text style={styles.patternDescription}>
                          Broader type: {item.description}
                        </Text>
                      ) : null}
                      <Text style={styles.patternExample}>
                        {item.recent_mistake_summary || "No recent example yet."}
                      </Text>
                    </View>
                    <Text style={styles.patternCount}>{item.count}</Text>
                  </View>
                ))
              )
            ) : null}
          </SectionCard>

          {allSpeakingWins.length > 0 ? (
            <SectionCard>
              <Pressable
                onPress={() => setWinsExpanded((prev) => !prev)}
                style={styles.expandHeader}
              >
                <Text style={styles.sectionTitle}>
                  Recent Improvement Wins ({allSpeakingWins.length})
                </Text>
                <Text style={styles.expandToggle}>{winsExpanded ? "Hide" : "Show"}</Text>
              </Pressable>
              {winsExpanded
                ? allSpeakingWins.map((win) => (
                    <View key={win.event_id}>
                      <Pressable
                        onPress={() =>
                          setSelectedSpeakingWinId((prev) =>
                            prev === win.event_id ? null : win.event_id
                          )
                        }
                        style={styles.winRow}
                      >
                        <Text style={styles.winSummary}>{win.summary}</Text>
                        <Text style={styles.winDate}>{win.created_at}</Text>
                      </Pressable>
                      {selectedSpeakingWinId === win.event_id ? (
                        <SpeakingWinDetails win={win} />
                      ) : null}
                    </View>
                  ))
                : null}
            </SectionCard>
          ) : null}

        </>
      )}
    </Screen>
  );
}

export default memo(InsightsScreen);

const styles = StyleSheet.create({
  title: {
    color: colors.text,
    fontSize: 28,
    fontWeight: "800",
    marginBottom: 16,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "800",
    marginBottom: 12,
    flex: 1,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 12,
  },
  muted: {
    color: colors.textSoft,
    fontSize: 14,
  },
  bannerList: {
    gap: 8,
    marginBottom: 16,
  },
  banner: {
    backgroundColor: colors.greenTint,
    borderColor: "#86efac",
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
  },
  bannerText: {
    color: "#166534",
    fontWeight: "700",
    fontSize: 14,
  },
  detailCard: {
    marginTop: 12,
    backgroundColor: colors.surface,
    borderColor: "#d1fae5",
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
  },
  metaLabel: {
    color: colors.textMuted,
    fontSize: 13,
    marginBottom: 6,
  },
  metaValue: {
    color: colors.text,
    fontWeight: "700",
  },
  metaDate: {
    color: colors.textSoft,
    fontSize: 12,
    marginBottom: 14,
  },
  detailHeadingBad: {
    color: "#991b1b",
    fontSize: 12,
    fontWeight: "800",
    marginBottom: 4,
  },
  detailHeadingGood: {
    color: "#166534",
    fontSize: 12,
    fontWeight: "800",
    marginTop: 12,
    marginBottom: 4,
  },
  detailText: {
    color: colors.text,
    lineHeight: 22,
  },
  expandHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
    gap: 12,
  },
  expandToggle: {
    color: colors.textSoft,
    fontWeight: "700",
  },
  winRow: {
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  winSummary: {
    color: colors.text,
    fontWeight: "700",
    marginBottom: 4,
  },
  winDate: {
    color: colors.textSoft,
    fontSize: 12,
  },
  patternRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 14,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  patternCopy: {
    flex: 1,
  },
  patternTitle: {
    color: colors.text,
    fontWeight: "700",
    marginBottom: 4,
  },
  patternDescription: {
    color: colors.textSoft,
    fontSize: 12,
    marginBottom: 6,
  },
  patternExample: {
    color: colors.textMuted,
    lineHeight: 20,
  },
  patternCount: {
    color: colors.primary,
    fontSize: 24,
    fontWeight: "800",
    minWidth: 34,
    textAlign: "right",
  },
  recentRow: {
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  recentLabel: {
    color: colors.text,
    fontWeight: "700",
    marginBottom: 4,
  },
  recentText: {
    color: colors.textMuted,
    lineHeight: 20,
  },
  recentMeta: {
    color: colors.textSoft,
    marginTop: 4,
    fontSize: 13,
  },
});
