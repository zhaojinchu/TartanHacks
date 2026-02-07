import type { BinMeasurement } from "../types";

type OnMeasurement = (measurement: BinMeasurement) => void;

export class BinWebSocketClient {
  private ws: WebSocket | null = null;

  connect(onMeasurement: OnMeasurement): void {
    const base = import.meta.env.VITE_WS_BASE_URL
      ? import.meta.env.VITE_WS_BASE_URL
      : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;

    this.ws = new WebSocket(`${base}/ws/bins`);

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "measurement") {
          onMeasurement(payload.data as BinMeasurement);
        }
      } catch {
        // Ignore malformed payloads.
      }
    };

    this.ws.onopen = () => {
      this.ws?.send("subscribe");
    };
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }
}
