import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { BinPrediction } from "../types";

interface Props {
  predictions: BinPrediction[];
}

export function PredictionChart({ predictions }: Props): JSX.Element {
  const data = predictions.map((item) => ({
    bin_id: item.bin_id,
    hours_to_target: item.hours_to_target ?? 0,
    fill_rate_per_hour: item.fill_rate_per_hour ?? 0
  }));

  return (
    <section className="panel">
      <h2>Time-to-Full Prediction (85%)</h2>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.12)" />
            <XAxis dataKey="bin_id" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="hours_to_target" fill="#f97316" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
