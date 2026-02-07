import { Fragment } from "react";

import type { HeatmapCell } from "../types";

interface Props {
  title: string;
  cells: HeatmapCell[];
}

function colorFor(value: number, max: number): string {
  if (max <= 0) {
    return "rgba(255,255,255,0.04)";
  }
  const ratio = Math.min(value / max, 1);
  const alpha = 0.08 + ratio * 0.78;
  return `rgba(16, 185, 129, ${alpha.toFixed(2)})`;
}

export function Heatmap({ title, cells }: Props): JSX.Element {
  if (cells.length === 0) {
    return (
      <section className="panel">
        <h2>{title}</h2>
        <p>No heatmap data yet.</p>
      </section>
    );
  }

  const xValues = Array.from(new Set(cells.map((cell) => cell.x)));
  const yValues = Array.from(new Set(cells.map((cell) => cell.y)));
  const maxValue = Math.max(...cells.map((cell) => cell.value));

  return (
    <section className="panel">
      <h2>{title}</h2>
      <div className="heatmap-grid" style={{ gridTemplateColumns: `120px repeat(${xValues.length}, minmax(20px, 1fr))` }}>
        <div className="heatmap-header" />
        {xValues.map((x) => (
          <div key={`x-${x}`} className="heatmap-header">
            {x}
          </div>
        ))}

        {yValues.map((y) => (
          <Fragment key={`row-${y}`}>
            <div key={`row-${y}`} className="heatmap-axis">
              {y}
            </div>
            {xValues.map((x) => {
              const cell = cells.find((item) => item.x === x && item.y === y);
              const value = cell?.value ?? 0;
              return (
                <div
                  key={`${x}-${y}`}
                  className="heatmap-cell"
                  title={`${y} / ${x}: ${value.toFixed(2)}`}
                  style={{ backgroundColor: colorFor(value, maxValue) }}
                >
                  {value > 0 ? value.toFixed(1) : ""}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </section>
  );
}
