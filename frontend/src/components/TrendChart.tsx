import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
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

interface Props {
  trends: TrendPoint[];
}

export default function TrendChart({ trends }: Props) {
  if (trends.length === 0) {
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
        No trend data yet. Analyze some sessions to see trends.
      </div>
    );
  }

  // Pivot: group by session date, with one key per mistake_type_code
  const sessionMap = new Map<
    string,
    Record<string, number> & { date: string }
  >();
  const allTypes = new Set<string>();

  for (const t of trends) {
    allTypes.add(t.mistake_type_code);
    const key = `${t.session_id}-${t.date}`;
    if (!sessionMap.has(key)) {
      sessionMap.set(key, { date: t.date });
    }
    const entry = sessionMap.get(key)!;
    entry[t.mistake_type_code] = t.count;
  }

  const chartData = Array.from(sessionMap.values());
  const typeList = Array.from(allTypes);

  return (
    <div
      style={{
        background: "#fff",
        borderRadius: 8,
        border: "1px solid #e2e8f0",
        padding: 16,
      }}
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" fontSize={12} />
          <YAxis fontSize={12} allowDecimals={false} />
          <Tooltip />
          <Legend />
          {typeList.map((code, i) => (
            <Line
              key={code}
              type="monotone"
              dataKey={code}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
