import React, { memo, useEffect, useMemo, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import Screen from "../components/Screen";
import SectionCard from "../components/SectionCard";
import VerticalBarChart from "../components/VerticalBarChart";
import PieChart from "../components/PieChart";
import TimeSeriesChart from "../components/TimeSeriesChart";
import { getInsights } from "../services/api";
import { useLanguageContext } from "../contexts/LanguageContext";
import { useLandingState } from "../contexts/LandingStateContext";
import type { InsightsResponse, SpeakingWinItem } from "../types";
import { colors } from "../theme";

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
  const [selectedSpeakingWinId, setSelectedSpeakingWinId] = useState<number | null>(null);

  useEffect(() => {
    if (!currentLanguageProfile) {
      if (!isLoadingLanguage) {
        setLoading(false);
      }
      return;
    }
    setLoading(true);
    getInsights(10, 30, currentLanguageProfile.id)
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

  const topMistakeItems = useMemo(
    () =>
      data?.top_mistakes.map((item, index) => ({
        key: `top-${item.code}-${index}`,
        label: item.label,
        value: item.count,
        accent: ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"][index % 5],
        helper: item.recent_mistake_summary || item.code,
      })) || [],
    [data]
  );

  const topMistakePie = useMemo(
    () =>
      data?.top_mistakes.slice(0, 5).map((item, index) => ({
        key: `pie-${item.code}-${index}`,
        label: item.label,
        value: item.count,
        color: ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"][index % 5],
      })) || [],
    [data]
  );

  const trendItems = useMemo(() => {
    if (!data) {
      return [];
    }
    const grouped = new Map<string, number>();
    data.trends.forEach((point) => {
      grouped.set(
        point.mistake_type_code,
        (grouped.get(point.mistake_type_code) || 0) + point.count
      );
    });
    return Array.from(grouped.entries()).map(([label, value], index) => ({
      key: `trend-${label}-${index}`,
      label,
      value,
      accent: ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"][index % 5],
    }));
  }, [data]);

  const progressItems = useMemo(
    () =>
      data?.progress.map((point, index) => ({
        key: `progress-${point.session_id}-${index}`,
        label: point.date.slice(5, 16),
        value: point.error_rate_per_100_words,
        helper: `${point.mistake_count} mistakes across ${point.word_count} words`,
      })) || [],
    [data]
  );

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
                    prev === data.latest_speaking_win?.event_id ? null : data.latest_speaking_win?.event_id ?? null
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

          {previousSpeakingWins.length > 0 ? (
            <SectionCard>
              <Pressable
                onPress={() => setWinsExpanded((prev) => !prev)}
                style={styles.expandHeader}
              >
                <Text style={styles.sectionTitle}>
                  Previous Speaking Wins ({previousSpeakingWins.length})
                </Text>
                <Text style={styles.expandToggle}>{winsExpanded ? "Hide" : "Show"}</Text>
              </Pressable>
              {winsExpanded
                ? previousSpeakingWins.map((win) => (
                    <View key={win.event_id}>
                      <Pressable
                        onPress={() =>
                          setSelectedSpeakingWinId((prev) => (prev === win.event_id ? null : win.event_id))
                        }
                        style={styles.winRow}
                      >
                        <Text style={styles.winSummary}>{win.summary}</Text>
                        <Text style={styles.winDate}>{win.created_at}</Text>
                      </Pressable>
                      {selectedSpeakingWinId === win.event_id ? <SpeakingWinDetails win={win} /> : null}
                    </View>
                  ))
                : null}
            </SectionCard>
          ) : null}

          <SectionCard>
            <Text style={styles.sectionTitle}>Top Mistake Types</Text>
            <VerticalBarChart
              data={topMistakeItems}
              emptyMessage="No mistakes recorded yet. Sessions with zero mistakes still appear in progress and history."
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
            <Text style={styles.sectionTitle}>Improvement Over Time</Text>
            <TimeSeriesChart
              data={progressItems}
              emptyMessage="No progress data yet."
            />
          </SectionCard>

          <SectionCard>
            <Text style={styles.sectionTitle}>Error Trends By Type</Text>
            <VerticalBarChart
              data={trendItems}
              emptyMessage="No trend data yet. Analyze some sessions to see trends."
              height={160}
            />
          </SectionCard>

          <SectionCard>
            <Text style={styles.sectionTitle}>Recent Mistakes</Text>
            {data.recent_mistakes.length === 0 ? (
              <Text style={styles.muted}>No recent mistakes yet.</Text>
            ) : (
              data.recent_mistakes.slice(0, 10).map((item) => (
                <View key={item.id} style={styles.recentRow}>
                  <Text style={styles.recentLabel}>{item.mistake_type_label}</Text>
                  <Text style={styles.recentText}>
                    {item.transcript_span || "(unknown)"} {"->"}{" "}
                    {item.suggested_correction || "(no suggestion)"}
                  </Text>
                  {item.explanation_short ? (
                    <Text style={styles.recentMeta}>{item.explanation_short}</Text>
                  ) : null}
                </View>
              ))
            )}
          </SectionCard>
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
