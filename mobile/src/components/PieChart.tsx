import React, { useMemo } from "react";
import { StyleSheet, Text, View } from "react-native";
import Svg, { Circle } from "react-native-svg";
import { colors } from "../theme";

interface Slice {
  key: string;
  label: string;
  value: number;
  color: string;
}

export default function PieChart({
  data,
  emptyMessage,
}: {
  data: Slice[];
  emptyMessage: string;
}) {
  const total = useMemo(
    () => data.reduce((sum, item) => sum + item.value, 0),
    [data]
  );

  if (!data.length || total <= 0) {
    return <Text style={styles.empty}>{emptyMessage}</Text>;
  }

  const radius = 56;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <View style={styles.row}>
      <Svg width={140} height={140} viewBox="0 0 140 140">
        <Circle
          cx="70"
          cy="70"
          r={radius}
          stroke={colors.surfaceMuted}
          strokeWidth="24"
          fill="none"
        />
        {data.map((slice) => {
          const fraction = slice.value / total;
          const dash = circumference * fraction;
          const node = (
            <Circle
              key={slice.key}
              cx="70"
              cy="70"
              r={radius}
              stroke={slice.color}
              strokeWidth="24"
              fill="none"
              strokeDasharray={`${dash} ${circumference - dash}`}
              strokeDashoffset={-offset}
              rotation="-90"
              origin="70,70"
              strokeLinecap="butt"
            />
          );
          offset += dash;
          return node;
        })}
      </Svg>
      <View style={styles.legend}>
        {data.map((slice) => (
          <View key={slice.key} style={styles.legendRow}>
            <View style={[styles.dot, { backgroundColor: slice.color }]} />
            <View style={styles.legendCopy}>
              <Text style={styles.legendLabel}>{slice.label}</Text>
              <Text style={styles.legendValue}>
                {slice.value} ({Math.round((slice.value / total) * 100)}%)
              </Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 16,
  },
  legend: {
    flex: 1,
    minWidth: 140,
    gap: 10,
  },
  legendRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 999,
  },
  legendCopy: {
    flex: 1,
  },
  legendLabel: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 13,
  },
  legendValue: {
    color: colors.textSoft,
    fontSize: 12,
  },
  empty: {
    color: colors.textSoft,
    fontSize: 14,
  },
});
