import { useEffect, useMemo, useState } from "react";

import { LiveGrid } from "../components/LiveGrid";
import { prettyBinNameFromType } from "../services/presentation";
import { getBins, markBinEmpty } from "../services/api";
import { BinWebSocketClient } from "../services/websocket";
import type { BinMeasurement } from "../types";

export function Dashboard(): JSX.Element {
  const [bins, setBins] = useState<BinMeasurement[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let mounted = true;
    const socket = new BinWebSocketClient();

    const refresh = async () => {
      try {
        const payload = await getBins();
        if (mounted) {
          setBins(payload);
          setError("");
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : "Failed to load bins");
        }
      }
    };

    refresh();
    const interval = window.setInterval(refresh, 10000);

    socket.connect((measurement) => {
      setBins((current) => {
        const next = [...current];
        const idx = next.findIndex((item) => item.bin_id === measurement.bin_id);
        if (idx >= 0) {
          next[idx] = { ...next[idx], ...measurement };
        } else {
          next.push(measurement);
        }
        return next;
      });
    });

    return () => {
      mounted = false;
      window.clearInterval(interval);
      socket.close();
    };
  }, []);

  const alerts = useMemo(
    () => bins.filter((bin) => (bin.fullness_percent ?? 0) >= 90).sort((a, b) => (b.fullness_percent ?? 0) - (a.fullness_percent ?? 0)),
    [bins]
  );

  const handleEmpty = async (binId: string) => {
    try {
      await markBinEmpty(binId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to mark bin empty");
    }
  };

  return (
    <main className="page">
      <section className="hero">
        <h1>Real-Time Bin Monitoring</h1>
        <p>Live ultrasonic measurements streamed from Raspberry Pi sensors.</p>
      </section>

      {error && <div className="error-banner">{error}</div>}

      {alerts.length > 0 && (
        <section className="alert-strip">
          <h2>Critical Alerts</h2>
          <div className="alert-list">
            {alerts.map((bin) => (
              <div className="alert-chip" key={bin.bin_id}>
                {prettyBinNameFromType(bin.bin_type)}: {bin.fullness_percent?.toFixed(1)}%
              </div>
            ))}
          </div>
        </section>
      )}

      <LiveGrid bins={bins} onEmpty={handleEmpty} />
    </main>
  );
}
