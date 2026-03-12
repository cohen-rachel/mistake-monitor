import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { colors } from "../theme";

interface Item {
  label: string;
  value: number;
  accent?: string;
  helper?: string;
}

export default function MiniBarChart({
  items,
  emptyMessage,
}: {
  items: Item[];
  emptyMessage: string;
}) {
  const max = Math.max(...items.map((item) => item.value), 1);

  if (items.length === 0) {
    return <Text style={styles.empty}>{emptyMessage}</Text>;
  }

  return (
    <View style={styles.container}>
      {items.map((item, index) => (
        <View key={`${item.label}-${item.helper || ""}-${index}`} style={styles.row}>
          <View style={styles.rowTop}>
            <Text style={styles.label}>{item.label}</Text>
            <Text style={styles.value}>{item.value}</Text>
          </View>
          <View style={styles.track}>
            <View
              style={[
                styles.fill,
                {
                  width: `${Math.max((item.value / max) * 100, 6)}%`,
                  backgroundColor: item.accent || colors.primary,
                },
              ]}
            />
          </View>
          {item.helper ? <Text style={styles.helper}>{item.helper}</Text> : null}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: 14,
  },
  row: {
    gap: 6,
  },
  rowTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 8,
  },
  label: {
    flex: 1,
    color: colors.text,
    fontSize: 14,
    fontWeight: "600",
  },
  value: {
    color: colors.textMuted,
    fontSize: 13,
    fontWeight: "700",
  },
  track: {
    height: 10,
    backgroundColor: colors.surfaceMuted,
    borderRadius: 999,
    overflow: "hidden",
  },
  fill: {
    height: "100%",
    borderRadius: 999,
  },
  helper: {
    color: colors.textSoft,
    fontSize: 12,
  },
  empty: {
    color: colors.textSoft,
    fontSize: 14,
  },
});
