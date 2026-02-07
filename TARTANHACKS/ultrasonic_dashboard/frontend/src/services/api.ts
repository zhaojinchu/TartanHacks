import type {
  BinMeasurement,
  BinPrediction,
  FillRateTrendPoint,
  FillTimeResponse,
  HeatmapResponse,
  PredictionAccuracyRow,
  ScheduleResponse
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Request failed (${response.status}): ${text}`);
  }

  return (await response.json()) as T;
}

export function getBins(): Promise<BinMeasurement[]> {
  return request<BinMeasurement[]>("/api/bins");
}

export function getBinHistory(binId: string, days = 7): Promise<{ items: BinMeasurement[]; count: number }> {
  const start = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
  return request(`/api/bins/${binId}/history?start=${encodeURIComponent(start)}&limit=2000`);
}

export function getPrediction(binId: string, targetFullness = 90): Promise<BinPrediction> {
  return request(`/api/bins/${binId}/prediction?target_fullness=${targetFullness}`);
}

export function getFillTimes(days = 30): Promise<FillTimeResponse> {
  return request(`/api/analytics/fill-times?days=${days}`);
}

export function getFillRateTrends(days = 30): Promise<FillRateTrendPoint[]> {
  return request(`/api/analytics/fill-rate-trends?days=${days}`);
}

export function getHeatmap(mode: "temporal" | "location", days = 30): Promise<HeatmapResponse> {
  return request(`/api/analytics/heatmap?mode=${mode}&days=${days}`);
}

export function getSchedule(targetFullness = 90): Promise<ScheduleResponse> {
  return request(`/api/schedule/optimize?target_fullness=${targetFullness}`);
}

export function getPredictionAccuracy(limit = 100): Promise<PredictionAccuracyRow[]> {
  return request(`/api/analytics/prediction-accuracy?limit=${limit}`);
}

export async function markBinEmpty(binId: string, reason = "manual_override"): Promise<void> {
  await request(`/api/bins/${binId}/empty`, {
    method: "POST",
    body: JSON.stringify({ reason })
  });
}
