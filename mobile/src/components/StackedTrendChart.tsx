import React, { useMemo, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import Svg, { G, Line, Rect, Text as SvgText } from "react-native-svg";
import type { TrendPoint } from "../types";
import { colors } from "../theme";

type TrendRange = "7d" | "4w" | "1y" | "all";

const PALETTE = [
  "#4338ca",
  "#0891b2",
  "#059669",
  "#d97706",
  "#dc2626",
  "#7c3aed",
  "#db2777",
  "#ea580c",
  "#0d9488",
  "#6366f1",
];

function parseTrendDate(value: string): Date {
  return new Date(value.replace(" ", "T"));
}

function startOfWeek(date: Date): Date {
  const next = new Date(date);
  const day = next.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  next.setDate(next.getDate() + diff);
  next.setHours(0, 0, 0, 0);
  return next;
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

function formatAxisDate(date: Date): string {
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

export default function StackedTrendChart({
  trends,
  range,
  emptyMessage,
}: {
  trends: TrendPoint[];
  range: TrendRange;
  emptyMessage: string;
}) {
  const [hiddenTypes, setHiddenTypes] = useState<string[]>([]);
  const chartWidth = 340;
  const chartHeight = 240;
  const leftPadding = 34;
  const rightPadding = 16;
  const topPadding = 18;
  const bottomPadding = 34;

  const filteredTrends = useMemo(() => {
    const cutoff = cutoffForRange(range);
    return trends.filter((item) => {
      if (!cutoff) {
        return true;
      }
      return parseTrendDate(item.date) >= cutoff;
    });
  }, [range, trends]);

  const { chartData, typeList, colorByCode, maxTotal } = useMemo(() => {
    const allTypes = new Set<string>();
    const grouped = new Map<
      string,
      {
        bucketKey: string;
        bucketDate: Date;
        hoverLabel: string;
        values: Record<string, number>;
      }
    >();
    const useWeeklyBuckets = range === "1y" || range === "all";

    for (const point of filteredTrends) {
      allTypes.add(point.mistake_type_code);
      const pointDate = parseTrendDate(point.date);
      const bucketDate = useWeeklyBuckets
        ? startOfWeek(pointDate)
        : new Date(pointDate.getFullYear(), pointDate.getMonth(), pointDate.getDate());
      const bucketKey = bucketDate.toISOString().slice(0, 10);

      if (!grouped.has(bucketKey)) {
        grouped.set(bucketKey, {
          bucketKey,
          bucketDate,
          hoverLabel: formatAxisDate(bucketDate),
          values: {},
        });
      }
      const row = grouped.get(bucketKey)!;
      row.values[point.mistake_type_code] =
        (row.values[point.mistake_type_code] || 0) + point.count;
    }

    const rows = Array.from(grouped.values()).sort(
      (a, b) => a.bucketDate.getTime() - b.bucketDate.getTime()
    );
    const codes = Array.from(allTypes);
    const maxTotal = Math.max(
      ...rows.map((row) => codes.reduce((sum, code) => sum + (row.values[code] || 0), 0)),
      1
    );

    return {
      chartData: rows,
      typeList: codes,
      colorByCode: new Map(codes.map((code, index) => [code, PALETTE[index % PALETTE.length]])),
      maxTotal,
    };
  }, [filteredTrends, range]);

  const visibleTypes = useMemo(
    () => typeList.filter((code) => !hiddenTypes.includes(code)),
    [hiddenTypes, typeList]
  );

  const yTicks = useMemo(() => {
    const tickCount = 4;
    return Array.from({ length: tickCount + 1 }, (_, index) => {
      const value = Math.round(maxTotal - (maxTotal * index) / tickCount);
      const y =
        topPadding + (index * (chartHeight - topPadding - bottomPadding)) / tickCount;
      return { key: `y-${index}`, value, y };
    });
  }, [maxTotal]);

  if (chartData.length === 0 || typeList.length === 0) {
    return <Text style={styles.empty}>{emptyMessage}</Text>;
  }

  const firstTick = chartData[0];
  const lastTick = chartData[chartData.length - 1];
  const barSlotWidth =
    (chartWidth - leftPadding - rightPadding) / Math.max(chartData.length, 1);
  const barWidth = Math.max(Math.min(barSlotWidth * 0.66, 28), 10);
  const plotHeight = chartHeight - topPadding - bottomPadding;

  const toggleType = (code: string) => {
    setHiddenTypes((prev) =>
      prev.includes(code) ? prev.filter((item) => item !== code) : [...prev, code]
    );
  };

  return (
    <View>
      <View style={styles.legend}>
        {typeList.map((code) => {
          const isHidden = hiddenTypes.includes(code);
          const color = colorByCode.get(code) || PALETTE[0];
          return (
            <Pressable
              key={code}
              onPress={() => toggleType(code)}
              style={styles.legendItem}
            >
              <View
                style={[
                  styles.legendDot,
                  { backgroundColor: color, opacity: isHidden ? 0.35 : 1 },
                ]}
              />
              <Text style={[styles.legendText, isHidden ? styles.legendTextHidden : null]}>
                {humanizeCode(code)}
              </Text>
            </Pressable>
          );
        })}
      </View>
      <Svg width="100%" height={chartHeight} viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
        {yTicks.map((tick) => (
          <G key={tick.key}>
            <Line
              x1={leftPadding}
              y1={tick.y}
              x2={chartWidth - rightPadding}
              y2={tick.y}
              stroke={colors.border}
              strokeWidth="1"
              strokeDasharray={tick.value === 0 ? undefined : "3 4"}
            />
            <SvgText
              x={leftPadding - 6}
              y={tick.y + 4}
              fontSize="10"
              fill={colors.textSoft}
              textAnchor="end"
            >
              {tick.value}
            </SvgText>
          </G>
        ))}
        <Line
          x1={leftPadding}
          y1={chartHeight - bottomPadding}
          x2={chartWidth - rightPadding}
          y2={chartHeight - bottomPadding}
          stroke={colors.border}
          strokeWidth="1.5"
        />
        {chartData.map((row, index) => {
          const x = leftPadding + index * barSlotWidth + (barSlotWidth - barWidth) / 2;
          let yCursor = chartHeight - bottomPadding;
          return (
            <G key={row.bucketKey}>
              {visibleTypes.map((code) => {
                const value = row.values[code] || 0;
                if (value <= 0) {
                  return null;
                }
                const height = Math.max((value / maxTotal) * plotHeight, 2);
                yCursor -= height;
                return (
                  <Rect
                    key={`${row.bucketKey}-${code}`}
                    x={x}
                    y={yCursor}
                    width={barWidth}
                    height={height}
                    fill={colorByCode.get(code) || PALETTE[0]}
                  />
                );
              })}
            </G>
          );
        })}
        {firstTick ? (
          <SvgText
            x={leftPadding}
            y={chartHeight - 12}
            fontSize="10"
            fill={colors.textSoft}
            textAnchor="start"
          >
            {firstTick.hoverLabel}
          </SvgText>
        ) : null}
        {lastTick && lastTick.bucketKey !== firstTick?.bucketKey ? (
          <SvgText
            x={chartWidth - rightPadding}
            y={chartHeight - 12}
            fontSize="10"
            fill={colors.textSoft}
            textAnchor="end"
          >
            {lastTick.hoverLabel}
          </SvgText>
        ) : null}
      </Svg>
      <Text style={styles.hint}>Tap a legend item to show or hide that error type.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  legend: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 10,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 999,
  },
  legendText: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "700",
  },
  legendTextHidden: {
    color: colors.textSoft,
  },
  hint: {
    color: colors.textSoft,
    fontSize: 12,
    marginTop: 4,
  },
  empty: {
    color: colors.textSoft,
    fontSize: 14,
  },
});
