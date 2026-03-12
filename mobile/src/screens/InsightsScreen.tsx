import React, { useEffect, useMemo, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import Screen from "../components/Screen";
import SectionCard from "../components/SectionCard";
import VerticalBarChart from "../components/VerticalBarChart";
import PieChart from "../components/PieChart";
import TimeSeriesChart from "../components/TimeSeriesChart";
import { getInsights } from "../services/api";
import { useLanguageContext } from "../contexts/LanguageContext";
import type { InsightsResponse } from "../types";
import { colors } from "../theme";

export default function InsightsScreen() {
  const { currentLanguageProfile, isLoadingLanguage } = useLanguageContext();
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [loading, setLoading] = useState(true);

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
  }, [currentLanguageProfile, isLoadingLanguage]);

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
          {data.improvement_banners.length > 0 ? (
            <View style={styles.bannerList}>
              {data.improvement_banners.map((banner, index) => (
                <View key={`${banner}-${index}`} style={styles.banner}>
                  <Text style={styles.bannerText}>{banner}</Text>
                </View>
              ))}
            </View>
          ) : null}

          <SectionCard>
            <Text style={styles.sectionTitle}>Top Mistake Types</Text>
            <VerticalBarChart
              data={topMistakeItems}
              emptyMessage="No mistakes recorded yet."
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
                    {item.transcript_span || "(unknown)"} ->{" "}
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
  note: {
    color: colors.textSoft,
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 10,
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
