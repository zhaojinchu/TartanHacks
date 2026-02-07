import { useEffect, useState } from "react";

import { Heatmap } from "../components/Heatmap";
import { getHeatmap } from "../services/api";
import type { HeatmapCell } from "../types";

export function Heatmaps(): JSX.Element {
  const [days, setDays] = useState<number>(30);
  const [temporalCells, setTemporalCells] = useState<HeatmapCell[]>([]);
  const [locationCells, setLocationCells] = useState<HeatmapCell[]>([]);

  useEffect(() => {
    Promise.all([getHeatmap("temporal", days), getHeatmap("location", days)]).then(([temporal, location]) => {
      setTemporalCells(temporal.cells);
      setLocationCells(location.cells);
    });
  }, [days]);

  return (
    <main className="page">
      <section className="hero">
        <h1>Heatmap Visualizations</h1>
        <p>Temporal and location intensity patterns for waste generation.</p>
        <div className="range-switch">
          {[7, 30, 90].map((value) => (
            <button key={value} className={days === value ? "solid-btn" : "ghost-btn"} onClick={() => setDays(value)}>
              Last {value}d
            </button>
          ))}
        </div>
      </section>

      <Heatmap title="Temporal Heatmap (Day vs Hour)" cells={temporalCells} />
      <Heatmap title="Location Heatmap" cells={locationCells} />
    </main>
  );
}
