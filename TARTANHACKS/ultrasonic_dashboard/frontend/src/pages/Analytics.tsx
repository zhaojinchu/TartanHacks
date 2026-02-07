import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { getFillRateTrends, getFillTimes } from "../services/api";
import type { FillRateTrendPoint, FillTimeResponse } from "../types";

export function Analytics(): JSX.Element {
  const [days, setDays] = useState<number>(30);
  const [fillTimes, setFillTimes] = useState<FillTimeResponse | null>(null);
  const [trends, setTrends] = useState<FillRateTrendPoint[]>([]);

  useEffect(() => {
    Promise.all([getFillTimes(days), getFillRateTrends(days)]).then(([times, trendPoints]) => {
      setFillTimes(times);
      setTrends(trendPoints);
    });
  }, [days]);

  const barData = useMemo(
    () =>
      fillTimes?.by_bin_type.map((item) => ({
        group: item.group_key,
        average_hours_to_85: item.average_hours_to_85 ?? 0
      })) ?? [],
    [fillTimes]
  );

  return (
    <main className="page">
      <section className="hero">
        <h1>Usage Analytics</h1>
        <p>Average fill times and fill rate trends across bins.</p>
        <div className="range-switch">
          {[7, 30, 90].map((value) => (
            <button key={value} className={days === value ? "solid-btn" : "ghost-btn"} onClick={() => setDays(value)}>
              Last {value}d
            </button>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Average Time to 85% Full (Hours)</h2>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.12)" />
              <XAxis dataKey="group" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="average_hours_to_85" fill="#0ea5e9" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h2>Fill Rate Trend (% per hour)</h2>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trends}>
              <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.12)" />
              <XAxis dataKey="timestamp" tickFormatter={(value) => new Date(value).toLocaleDateString()} />
              <YAxis />
              <Tooltip labelFormatter={(value) => new Date(String(value)).toLocaleString()} />
              <Legend />
              <Line type="monotone" dataKey="fill_rate_per_hour" stroke="#f97316" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </main>
  );
}
