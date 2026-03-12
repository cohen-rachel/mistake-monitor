import React from "react";
import { StyleSheet, Text, View } from "react-native";
import type { MistakeOut } from "../types";
import { colors } from "../theme";

export default function MistakeCard({ mistake }: { mistake: MistakeOut }) {
  return (
    <View style={styles.card}>
      <View style={styles.topRow}>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{mistake.mistake_type.label}</Text>
        </View>
        {mistake.confidence != null && (
          <Text style={styles.metaText}>
            Confidence {Math.round(mistake.confidence * 100)}%
          </Text>
        )}
      </View>
      <View style={styles.topRow}>
        {mistake.stt_uncertain && (
          <View style={[styles.badge, styles.warnBadge]}>
            <Text style={[styles.badgeText, styles.warnBadgeText]}>STT uncertain</Text>
          </View>
        )}
        {mistake.uncertain && (
          <View style={[styles.badge, styles.warnBadge]}>
            <Text style={[styles.badgeText, styles.warnBadgeText]}>Uncertain</Text>
          </View>
        )}
      </View>
      <Text style={styles.line}>
        <Text style={styles.wrong}>{mistake.transcript_span || "(unknown)"}</Text>
        {mistake.suggested_correction ? (
          <Text>
            {"  ->  "}
            <Text style={styles.correct}>{mistake.suggested_correction}</Text>
          </Text>
        ) : null}
      </Text>
      {mistake.explanation_short ? (
        <Text style={styles.explanation}>{mistake.explanation_short}</Text>
      ) : null}
      {mistake.uncertain_reason ? (
        <Text style={styles.note}>Note: {mistake.uncertain_reason}</Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },
  topRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    alignItems: "center",
    marginBottom: 8,
  },
  badge: {
    backgroundColor: colors.violetTint,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  badgeText: {
    color: "#4338ca",
    fontWeight: "700",
    fontSize: 12,
  },
  warnBadge: {
    backgroundColor: colors.yellowTint,
  },
  warnBadgeText: {
    color: "#92400e",
  },
  metaText: {
    color: colors.textSoft,
    fontSize: 12,
  },
  line: {
    color: colors.text,
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 8,
  },
  wrong: {
    backgroundColor: colors.redTint,
    color: "#b91c1c",
    fontWeight: "600",
  },
  correct: {
    backgroundColor: colors.greenTint,
    color: "#166534",
    fontWeight: "600",
  },
  explanation: {
    color: colors.textMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  note: {
    marginTop: 6,
    color: "#92400e",
    fontSize: 13,
  },
});
