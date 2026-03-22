import React, { useMemo, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import Svg, {
  Circle,
  Line,
  Path,
  Rect,
  Text as SvgText,
  G,
} from "react-native-svg";
import { colors } from "../theme";

interface Point {
  key: string;
  label: string;
  value: number;
  helper?: string;
}

function buildPath(points: Array<{ x: number; y: number }>) {
  if (!points.length) {
    return "";
  }
  return points
    .map((point, index) =>
      `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`
    )
    .join(" ");
}

export default function TimeSeriesChart({
  data,
  emptyMessage,
  color = colors.danger,
  yAxisTitle = "Error Rate / 100 Words",
}: {
  data: Point[];
  emptyMessage: string;
  color?: string;
  yAxisTitle?: string;
}) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const chartWidth = 320;
  const chartHeight = 220;
  const leftPadding = 54;
  const rightPadding = 16;
  const topPadding = 18;
  const bottomPadding = 42;

  const normalized = useMemo(() => {
    if (!data.length) {
      return [];
    }
    const max = Math.max(...data.map((item) => item.value), 1);
    const min = Math.min(...data.map((item) => item.value), 0);
    const range = max - min || 1;
    return data.map((item, index) => {
      const x =
        leftPadding +
        (index * (chartWidth - leftPadding - rightPadding)) /
          Math.max(data.length - 1, 1);
      const y =
        chartHeight -
        bottomPadding -
        ((item.value - min) / range) *
          (chartHeight - topPadding - bottomPadding);
      return { ...item, x, y };
    });
  }, [data]);

  const { minValue, maxValue, yTicks, xTicks, selectedPoint } = useMemo(() => {
    const maxValue = Math.max(...data.map((item) => item.value), 1);
    const minValue = Math.min(...data.map((item) => item.value), 0);
    const tickCount = 4;
    const range = maxValue - minValue || 1;
    const yTicks = Array.from({ length: tickCount + 1 }, (_, index) => {
      const ratio = index / tickCount;
      const value = maxValue - range * ratio;
      const y =
        topPadding +
        (index * (chartHeight - topPadding - bottomPadding)) / tickCount;
      return {
        key: `y-${index}`,
        value,
        y,
      };
    });

    const xTickIndexes = Array.from(
      new Set([0, Math.floor((data.length - 1) / 2), Math.max(data.length - 1, 0)])
    );
    const xTicks = xTickIndexes
      .map((index) => normalized[index])
      .filter(Boolean)
      .map((point, index) => ({
        key: `x-${point.key}-${index}`,
        label: point.label,
        x: point.x,
      }));

    return {
      minValue,
      maxValue,
      yTicks,
      xTicks,
      selectedPoint:
        selectedKey
          ? normalized.find((point) => point.key === selectedKey) ?? null
          : null,
    };
  }, [data, normalized, selectedKey]);

  if (!data.length) {
    return <Text style={styles.empty}>{emptyMessage}</Text>;
  }

  const path = buildPath(normalized);

  return (
    <View>
      <Svg
        width="100%"
        height={chartHeight}
        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
      >
        {yTicks.map((tick) => (
          <G key={tick.key}>
            <Line
              x1={leftPadding}
              y1={tick.y}
              x2={chartWidth - rightPadding}
              y2={tick.y}
              stroke={colors.border}
              strokeWidth="1"
              strokeDasharray={tick.value === minValue ? undefined : "3 4"}
            />
            <SvgText
              x={leftPadding - 8}
              y={tick.y + 4}
              fontSize="10"
              fill={colors.textSoft}
              textAnchor="end"
            >
              {tick.value.toFixed(1)}
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
        <Line
          x1={leftPadding}
          y1={topPadding}
          x2={leftPadding}
          y2={chartHeight - bottomPadding}
          stroke={colors.border}
          strokeWidth="1.5"
        />
        <SvgText
          x={16}
          y={chartHeight / 2}
          fontSize="10"
          fill={colors.textSoft}
          rotation="-90"
          origin={`${16},${chartHeight / 2}`}
          textAnchor="middle"
        >
          {yAxisTitle}
        </SvgText>
        <Path d={path} fill="none" stroke={color} strokeWidth="3" />
        {normalized.map((point) => {
          const isSelected = selectedPoint?.key === point.key;
          return (
            <G key={point.key}>
              <Rect
                x={point.x - 18}
                y={topPadding}
                width={36}
                height={chartHeight - topPadding - bottomPadding}
                fill="transparent"
                onPressIn={() => setSelectedKey(point.key)}
                onPressOut={() => setSelectedKey(null)}
              />
              <Circle
                cx={point.x}
                cy={point.y}
                r={isSelected ? 5 : 4}
                fill={isSelected ? colors.primary : color}
              />
            </G>
          );
        })}
        {selectedPoint ? (
          <G>
            <Line
              x1={selectedPoint.x}
              y1={topPadding}
              x2={selectedPoint.x}
              y2={chartHeight - bottomPadding}
              stroke={colors.primary}
              strokeWidth="1"
              strokeDasharray="4 4"
            />
            <Rect
              x={Math.max(leftPadding, selectedPoint.x - 44)}
              y={Math.max(topPadding, selectedPoint.y - 38)}
              width={88}
              height={28}
              rx={8}
              fill={colors.ink}
            />
            <SvgText
              x={selectedPoint.x}
              y={Math.max(topPadding + 18, selectedPoint.y - 20)}
              fontSize="10"
              fill={colors.white}
              textAnchor="middle"
            >
              {selectedPoint.value.toFixed(2)}
            </SvgText>
          </G>
        ) : null}
        {xTicks.map((tick) => (
          <SvgText
            key={tick.key}
            x={tick.x}
            y={chartHeight - 12}
            fontSize="10"
            fill={colors.textSoft}
            textAnchor="middle"
          >
            {tick.label}
          </SvgText>
        ))}
        <SvgText
          x={(leftPadding + chartWidth - rightPadding) / 2}
          y={chartHeight - 2}
          fontSize="10"
          fill={colors.textSoft}
          textAnchor="middle"
        >
          Time
        </SvgText>
      </Svg>
      {selectedPoint?.helper ? (
        <Text style={styles.helper}>
          {selectedPoint.label}: {selectedPoint.helper}
        </Text>
      ) : (
        <Text style={styles.hint}>Press and hold a point to inspect it.</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  helper: {
    marginTop: 8,
    color: colors.textMuted,
    fontSize: 12,
  },
  hint: {
    marginTop: 8,
    color: colors.textSoft,
    fontSize: 12,
  },
  empty: {
    color: colors.textSoft,
    fontSize: 14,
  },
});
