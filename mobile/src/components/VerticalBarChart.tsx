import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { colors } from "../theme";

interface Datum {
  key: string;
  label: string;
  value: number;
  accent?: string;
}

export default function VerticalBarChart({
  data,
  emptyMessage,
  height = 180,
}: {
  data: Datum[];
  emptyMessage: string;
  height?: number;
}) {
  if (data.length === 0) {
    return <Text style={styles.empty}>{emptyMessage}</Text>;
  }

  const max = Math.max(...data.map((item) => item.value), 1);

  return (
    <View>
      <View style={[styles.chartArea, { height }]}>
        {data.map((item) => (
          <View key={item.key} style={styles.barColumn}>
            <View style={styles.barArea}>
              <Text style={styles.valueLabel}>{item.value}</Text>
              <View style={styles.track}>
                <View
                  style={[
                    styles.bar,
                    {
                      height: `${Math.max((item.value / max) * 100, 6)}%`,
                      backgroundColor: item.accent || colors.primary,
                    },
                  ]}
                />
              </View>
            </View>
            <View style={styles.labelBox}>
              <Text style={styles.axisLabel} numberOfLines={3}>
                {item.label}
              </Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  chartArea: {
    flexDirection: "row",
    alignItems: "stretch",
    justifyContent: "space-between",
    gap: 10,
  },
  barColumn: {
    flex: 1,
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  barArea: {
    flex: 1,
    width: "100%",
    alignItems: "center",
    justifyContent: "flex-end",
    gap: 8,
  },
  valueLabel: {
    color: colors.textSoft,
    fontSize: 12,
    fontWeight: "700",
  },
  track: {
    width: "100%",
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: colors.surfaceMuted,
    borderRadius: 12,
    overflow: "hidden",
    minHeight: 110,
  },
  bar: {
    width: "100%",
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    minHeight: 8,
  },
  labelBox: {
    width: "100%",
    minHeight: 44,
    justifyContent: "flex-start",
  },
  axisLabel: {
    color: colors.textMuted,
    fontSize: 11,
    lineHeight: 14,
    textAlign: "center",
    flexShrink: 1,
  },
  empty: {
    color: colors.textSoft,
    fontSize: 14,
  },
});
