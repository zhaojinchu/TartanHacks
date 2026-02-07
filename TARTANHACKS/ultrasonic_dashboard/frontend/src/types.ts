export type BinType = "recycle" | "compost" | "landfill";
export type BinStatus = "normal" | "almost_full" | "full" | "sensor_error";

export interface BinMeasurement {
  bin_id: string;
  bin_type: BinType;
  timestamp: string;
  distance_cm: number | null;
  fullness_percent: number | null;
  bin_height_cm: number;
  status: BinStatus;
  location: string;
  alerts_active?: boolean;
}

export interface FillTimeStat {
  group_key: string;
  count_cycles: number;
  average_hours_to_85: number | null;
}

export interface FillTimeResponse {
  by_bin_type: FillTimeStat[];
  by_location: FillTimeStat[];
  window_days: number;
}

export interface FillRateTrendPoint {
  bin_id: string;
  timestamp: string;
  fill_rate_per_hour: number;
}

export interface HeatmapCell {
  x: string;
  y: string;
  value: number;
}

export interface HeatmapResponse {
  mode: "temporal" | "location";
  window_days: number;
  cells: HeatmapCell[];
}

export interface BinPrediction {
  bin_id: string;
  target_fullness: number;
  current_fullness: number | null;
  predicted_full_at: string | null;
  hours_to_target: number | null;
  fill_rate_per_hour: number | null;
  confidence_low_hours: number | null;
  confidence_high_hours: number | null;
  confidence_score: number;
  anomalies: string[];
}

export interface ScheduleItem {
  bin_id: string;
  location: string;
  priority: number;
  predicted_full_at: string | null;
  eta_window: string;
}

export interface ScheduleResponse {
  generated_at: string;
  target_fullness: number;
  route: ScheduleItem[];
}

export interface PredictionAccuracyRow {
  id: number;
  bin_id: string;
  timestamp: string;
  target_fullness: number;
  predicted_full_at: string | null;
  confidence_low_hours: number | null;
  confidence_high_hours: number | null;
}
