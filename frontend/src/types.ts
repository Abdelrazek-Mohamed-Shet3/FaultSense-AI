export type Lang = "en" | "ar";
export type SeverityLevel = "pass" | "warning" | "critical";

export interface HealthInfo {
  status: string;
  device: string;
  active_model: "builtin" | "custom";
  has_custom_model: boolean;
  custom_classes: string[];
  builtin_classes: number;
}

export interface SampleItem {
  id: string;
  filename: string;
  media_type: "image" | "video";
  domain: string;
  label_en: string;
  label_ar: string;
  severity_level: SeverityLevel;
  url: string;
}

export interface FaultRegion {
  id: number;
  x_norm: number;
  y_norm: number;
  w_norm: number;
  h_norm: number;
  peak_activation: number;
  area_pct: number;
}

export interface ClassProb {
  class_id: string;
  label_en: string;
  label_ar: string;
  domain: string;
  probability: number;
  severity_level: SeverityLevel;
  color: string;
}

export interface TimelinePoint {
  sample_index: number;
  frame_index: number;
  timestamp_sec: number;
  predicted_class: string;
  label_en: string;
  label_ar: string;
  is_fault: boolean;
  confidence: number;
  target_class_prob: number;
  temporal_attention: number;
  severity_score: number;
}

export interface KeyframeItem {
  rank: number;
  frame_index: number;
  timestamp_sec: number;
  attention_weight: number;
  confidence: number;
  regions: FaultRegion[];
  original_url: string;
  gradcam_url: string;
  localized_url: string;
}

export interface InspectionResult {
  id: string;
  created_at?: string;
  filename?: string;
  media_type: "image" | "video";
  model_mode: "builtin" | "custom";
  domain_filter: string;
  predicted_class: string;
  domain: string;
  label_en: string;
  label_ar: string;
  is_fault: boolean;
  severity_level: SeverityLevel;
  severity_score: number;
  confidence: number;
  description_en: string;
  description_ar: string;
  recommendation_en: string;
  recommendation_ar: string;
  regions: FaultRegion[];
  probabilities: ClassProb[];
  video_stats?: {
    fps: number;
    total_frames: number;
    duration_sec: number;
    sampled_frames: number;
    peak_timestamp_sec: number;
    peak_frame_index: number;
  };
  timeline?: TimelinePoint[];
  keyframes?: KeyframeItem[];
  assets: {
    original: string;
    gradcam: string;
    localized: string;
    annotated_video?: string;
  };
}

export interface ModelMetrics {
  available: boolean;
  device: string;
  backbone: string;
  num_classes: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  training_seconds: number;
  test_accuracy: number;
  test_f1_macro: number;
  test_precision_macro: number;
  test_recall_macro: number;
  video_val_accuracy: number;
  spatial_history: Array<{
    epoch: number;
    train_loss: number;
    train_acc: number;
    val_loss: number;
    val_acc: number;
    val_f1: number;
  }>;
  temporal_history: Array<{
    epoch: number;
    train_loss: number;
    train_acc: number;
    val_acc: number;
  }>;
  per_class: Array<{
    id: string;
    label_en: string;
    label_ar: string;
    domain: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }>;
  confusion_matrix: number[][];
  class_ids: string[];
  class_labels: string[];
}

export interface TrainingStatus {
  state: "idle" | "running" | "completed" | "error";
  epoch: number;
  total_epochs: number;
  best_val_acc: number;
  history: Array<{
    epoch: number;
    train_loss: number;
    train_acc: number;
    val_acc: number;
  }>;
  class_names: string[];
  error: string | null;
}
