import { useEffect, useState } from "react";

import { PredictionChart } from "../components/PredictionChart";
import { ScheduleView } from "../components/ScheduleView";
import { getBins, getPrediction, getPredictionAccuracy, getSchedule } from "../services/api";
import type { BinPrediction, PredictionAccuracyRow, ScheduleResponse } from "../types";

export function Predictions(): JSX.Element {
  const [predictions, setPredictions] = useState<BinPrediction[]>([]);
  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [accuracyRows, setAccuracyRows] = useState<PredictionAccuracyRow[]>([]);

  useEffect(() => {
    const load = async () => {
      const bins = await getBins();
      const nextPredictions = await Promise.all(bins.map((bin) => getPrediction(bin.bin_id)));
      const nextSchedule = await getSchedule(85);
      const logs = await getPredictionAccuracy(30);

      setPredictions(nextPredictions);
      setSchedule(nextSchedule);
      setAccuracyRows(logs);
    };

    load();
    const interval = window.setInterval(load, 45000);
    return () => window.clearInterval(interval);
  }, []);

  return (
    <main className="page">
      <section className="hero">
        <h1>Predictive Scheduling</h1>
        <p>Forecasted full times, optimized collection route, and prediction history.</p>
      </section>

      <PredictionChart predictions={predictions} />

      <section className="panel">
        <h2>Prediction Details</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Bin</th>
                <th>Current Fullness</th>
                <th>Rate (%/hr)</th>
                <th>Predicted 85% Time</th>
                <th>Confidence</th>
                <th>Anomalies</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((item) => (
                <tr key={item.bin_id}>
                  <td>{item.bin_id}</td>
                  <td>{item.current_fullness !== null ? `${item.current_fullness.toFixed(1)}%` : "N/A"}</td>
                  <td>{item.fill_rate_per_hour !== null ? item.fill_rate_per_hour.toFixed(2) : "N/A"}</td>
                  <td>{item.predicted_full_at ? new Date(item.predicted_full_at).toLocaleString() : "N/A"}</td>
                  <td>
                    {item.confidence_low_hours !== null && item.confidence_high_hours !== null
                      ? `${item.confidence_low_hours.toFixed(1)}h - ${item.confidence_high_hours.toFixed(1)}h`
                      : "N/A"}
                  </td>
                  <td>{item.anomalies.length > 0 ? item.anomalies.join(", ") : "none"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <ScheduleView route={schedule?.route ?? []} />

      <section className="panel">
        <h2>Historical Prediction Logs</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Bin</th>
                <th>Target</th>
                <th>Predicted Full At</th>
                <th>Confidence Hours</th>
              </tr>
            </thead>
            <tbody>
              {accuracyRows.map((row) => (
                <tr key={row.id}>
                  <td>{new Date(row.timestamp).toLocaleString()}</td>
                  <td>{row.bin_id}</td>
                  <td>{row.target_fullness}%</td>
                  <td>{row.predicted_full_at ? new Date(row.predicted_full_at).toLocaleString() : "N/A"}</td>
                  <td>
                    {row.confidence_low_hours !== null && row.confidence_high_hours !== null
                      ? `${row.confidence_low_hours.toFixed(1)} - ${row.confidence_high_hours.toFixed(1)}`
                      : "N/A"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
