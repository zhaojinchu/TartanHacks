import type { BinMeasurement } from "../types";

interface Props {
  bin: BinMeasurement;
  onEmpty: (binId: string) => void;
}

const statusLabels: Record<string, string> = {
  normal: "Normal",
  almost_full: "Almost Full",
  full: "Full",
  sensor_error: "Sensor Error"
};

function statusTone(status: BinMeasurement["status"]): string {
  switch (status) {
    case "normal":
      return "status-ok";
    case "almost_full":
      return "status-warn";
    case "full":
      return "status-danger";
    default:
      return "status-muted";
  }
}

export function BinCard({ bin, onEmpty }: Props): JSX.Element {
  const fullness = bin.fullness_percent ?? 0;
  const progressStyle = {
    background: `conic-gradient(var(--accent) ${Math.round(fullness * 3.6)}deg, rgba(255,255,255,0.1) 0deg)`
  };

  return (
    <article className="bin-card">
      <header className="bin-card-head">
        <div>
          <h3>{bin.bin_id}</h3>
          <p>{bin.bin_type.toUpperCase()}</p>
        </div>
        <span className={`status-pill ${statusTone(bin.status)}`}>{statusLabels[bin.status]}</span>
      </header>

      <div className="gauge-wrap">
        <div className="gauge" style={progressStyle}>
          <div className="gauge-inner">{bin.fullness_percent !== null ? `${fullness.toFixed(1)}%` : "--"}</div>
        </div>
      </div>

      <dl className="bin-meta">
        <div>
          <dt>Location</dt>
          <dd>{bin.location}</dd>
        </div>
        <div>
          <dt>Distance</dt>
          <dd>{bin.distance_cm !== null ? `${bin.distance_cm.toFixed(1)} cm` : "N/A"}</dd>
        </div>
        <div>
          <dt>Last Update</dt>
          <dd>{new Date(bin.timestamp).toLocaleString()}</dd>
        </div>
      </dl>

      <button type="button" className="ghost-btn" onClick={() => onEmpty(bin.bin_id)}>
        Mark Emptied
      </button>
    </article>
  );
}
