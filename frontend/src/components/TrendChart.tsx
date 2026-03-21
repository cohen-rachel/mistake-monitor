import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useMemo, useState } from "react";
import type { TrendPoint } from "../types";

const COLORS = [
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

export type TrendRange = "7d" | "4w" | "1y" | "all";

interface Props {
  trends: TrendPoint[];
  range: TrendRange;
}

type ChartRow = {
  bucketKey: string;
  bucketDate: Date;
  hoverLabel: string;
  [key: string]: string | number | Date;
};

export function parseTrendDate(value: string): Date {
  return new Date(value.replace(" ", "T"));
}

export function startOfTrendWeek(date: Date): Date {
  const next = new Date(date);
  const day = next.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  next.setDate(next.getDate() + diff);
  next.setHours(0, 0, 0, 0);
  return next;
}

export function formatAxisDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(date);
}

export function formatHoverDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "long",
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

export function cutoffForRange(range: TrendRange): Date | null {
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

export default function TrendChart({ trends, range }: Props) {
  const [hiddenTypes, setHiddenTypes] = useState<string[]>([]);

  const filteredTrends = useMemo(() => {
    const cutoff = cutoffForRange(range);
    return trends.filter((item) => {
      if (!cutoff) {
        return true;
      }
      return parseTrendDate(item.date) >= cutoff;
    });
  }, [range, trends]);

  const { chartData, typeList, colorByCode } = useMemo(() => {
    const allTypes = new Set<string>();
    const grouped = new Map<string, ChartRow>();
    const useWeeklyBuckets = range === "1y" || range === "all";

    for (const point of filteredTrends) {
      allTypes.add(point.mistake_type_code);
      const pointDate = parseTrendDate(point.date);
      const bucketDate = useWeeklyBuckets ? startOfTrendWeek(pointDate) : new Date(pointDate.getFullYear(), pointDate.getMonth(), pointDate.getDate());
      const bucketKey = bucketDate.toISOString().slice(0, 10);
      if (!grouped.has(bucketKey)) {
        grouped.set(bucketKey, {
          bucketKey,
          bucketDate,
          hoverLabel: formatHoverDate(bucketDate),
        });
      }
      const row = grouped.get(bucketKey)!;
      row[point.mistake_type_code] =
        Number(row[point.mistake_type_code] || 0) + point.count;
    }

    const rows = Array.from(grouped.values()).sort(
      (a, b) => a.bucketDate.getTime() - b.bucketDate.getTime()
    );
    const codes = Array.from(allTypes);
    for (const row of rows) {
      for (const code of codes) {
        if (typeof row[code] !== "number") {
          row[code] = 0;
        }
      }
    }
    return {
      chartData: rows,
      typeList: codes,
      colorByCode: new Map(codes.map((code, index) => [code, COLORS[index % COLORS.length]])),
    };
  }, [filteredTrends, range]);

  const visibleTypes = useMemo(
    () => typeList.filter((code) => !hiddenTypes.includes(code)),
    [hiddenTypes, typeList]
  );

  if (chartData.length === 0 || typeList.length === 0) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: 40,
          color: "#94a3b8",
          background: "#fff",
          borderRadius: 8,
          border: "1px solid #e2e8f0",
        }}
      >
        No trend data yet in this time range.
      </div>
    );
  }

  const firstTick = chartData[0]?.bucketKey;
  const lastTick = chartData[chartData.length - 1]?.bucketKey;

  const toggleType = (code: string) => {
    setHiddenTypes((prev) =>
      prev.includes(code) ? prev.filter((item) => item !== code) : [...prev, code]
    );
  };

  return (
    <div
      style={{
        background: "#fff",
        borderRadius: 8,
        border: "1px solid #e2e8f0",
        padding: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          marginBottom: 12,
        }}
      >
        {typeList.map((code) => {
          const isHidden = hiddenTypes.includes(code);
          const color = colorByCode.get(code) || COLORS[0];
          return (
            <button
              key={code}
              type="button"
              onClick={() => toggleType(code)}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                border: "none",
                background: "transparent",
                cursor: "pointer",
                padding: 0,
                color: isHidden ? "#94a3b8" : "#334155",
                fontSize: 12,
                fontWeight: 600,
              }}
            >
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 999,
                  background: color,
                  opacity: isHidden ? 0.35 : 1,
                }}
              />
              <span>{humanizeCode(code)}</span>
            </button>
          );
        })}
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="bucketKey"
            ticks={firstTick === lastTick ? [firstTick] : [firstTick, lastTick].filter(Boolean)}
            tickFormatter={(value) => formatAxisDate(new Date(`${value}T00:00:00`))}
            fontSize={12}
          />
          <YAxis fontSize={12} allowDecimals={false} />
          <Tooltip
            formatter={(value: number, name: string) => [value, humanizeCode(name)]}
            labelFormatter={(_, payload) => {
              const row = payload?.[0]?.payload as ChartRow | undefined;
              return row?.hoverLabel || "";
            }}
          />
          {visibleTypes.map((code) => (
            <Bar
              key={code}
              dataKey={code}
              stackId="errors"
              fill={colorByCode.get(code) || COLORS[0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
