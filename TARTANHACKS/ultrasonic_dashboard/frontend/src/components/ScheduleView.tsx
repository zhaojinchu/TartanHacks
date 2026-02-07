import type { ScheduleItem } from "../types";

interface Props {
  route: ScheduleItem[];
}

function downloadCsv(route: ScheduleItem[]): void {
  const header = ["priority", "bin_id", "location", "predicted_full_at", "eta_window"];
  const rows = route.map((item) => [
    item.priority,
    item.bin_id,
    item.location,
    item.predicted_full_at ?? "",
    item.eta_window
  ]);

  const csv = [header, ...rows].map((line) => line.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "optimized_schedule.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function exportPdf(route: ScheduleItem[]): void {
  if (route.length === 0) {
    return;
  }

  const rows = route
    .map(
      (item) =>
        `<tr>
          <td>${item.priority}</td>
          <td>${item.bin_id}</td>
          <td>${item.location}</td>
          <td>${item.predicted_full_at ? new Date(item.predicted_full_at).toLocaleString() : "N/A"}</td>
          <td>${item.eta_window}</td>
        </tr>`
    )
    .join("");

  const html = `<!doctype html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>Optimized Cleaner Schedule</title>
    <style>
      body { font-family: sans-serif; padding: 20px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #d1d5db; padding: 8px; text-align: left; font-size: 12px; }
      th { background: #f3f4f6; }
    </style>
  </head>
  <body>
    <h1>Optimized Cleaner Schedule</h1>
    <table>
      <thead>
        <tr><th>Priority</th><th>Bin</th><th>Location</th><th>Predicted Full</th><th>ETA Window</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  </body>
</html>`;

  const popup = window.open("", "_blank", "noopener,noreferrer");
  if (!popup) {
    return;
  }
  popup.document.open();
  popup.document.write(html);
  popup.document.close();
  popup.focus();
  popup.print();
}

export function ScheduleView({ route }: Props): JSX.Element {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Optimized Cleaner Schedule</h2>
        <div className="range-switch">
          <button type="button" className="ghost-btn" onClick={() => downloadCsv(route)} disabled={route.length === 0}>
            Export CSV
          </button>
          <button type="button" className="ghost-btn" onClick={() => exportPdf(route)} disabled={route.length === 0}>
            Export PDF
          </button>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Priority</th>
              <th>Bin</th>
              <th>Location</th>
              <th>Predicted Full</th>
              <th>ETA Window</th>
            </tr>
          </thead>
          <tbody>
            {route.map((item) => (
              <tr key={`${item.bin_id}-${item.priority}`}>
                <td>{item.priority}</td>
                <td>{item.bin_id}</td>
                <td>{item.location}</td>
                <td>{item.predicted_full_at ? new Date(item.predicted_full_at).toLocaleString() : "N/A"}</td>
                <td>{item.eta_window}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
