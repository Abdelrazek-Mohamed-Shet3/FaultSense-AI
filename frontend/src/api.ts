import type {
  HealthInfo,
  InspectionResult,
  ModelMetrics,
  SampleItem,
  TrainingStatus,
} from "./types";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => jsonFetch<HealthInfo>("/api/health"),
  samples: () => jsonFetch<{ images: SampleItem[]; videos: SampleItem[] }>("/api/samples"),
  metrics: () => jsonFetch<ModelMetrics>("/api/model/metrics"),
  history: (domain = "all", severity = "all") =>
    jsonFetch<{ items: InspectionResult[]; count: number }>(
      `/api/history?domain=${encodeURIComponent(domain)}&severity=${encodeURIComponent(severity)}`
    ),
  deleteHistory: (id: string) =>
    jsonFetch<{ deleted: boolean }>(`/api/history/${encodeURIComponent(id)}`, { method: "DELETE" }),

  inspectImage: async (opts: {
    file?: File;
    sampleId?: string;
    domainFilter: string;
    activeModel: string;
  }) => {
    const fd = new FormData();
    if (opts.file) fd.append("file", opts.file);
    if (opts.sampleId) fd.append("sample_id", opts.sampleId);
    fd.append("domain_filter", opts.domainFilter);
    fd.append("active_model", opts.activeModel);
    return jsonFetch<InspectionResult>("/api/inspect/image", { method: "POST", body: fd });
  },

  inspectVideo: async (opts: {
    file?: File;
    sampleId?: string;
    domainFilter: string;
    activeModel: string;
    sampleFrames?: number;
  }) => {
    const fd = new FormData();
    if (opts.file) fd.append("file", opts.file);
    if (opts.sampleId) fd.append("sample_id", opts.sampleId);
    fd.append("domain_filter", opts.domainFilter);
    fd.append("active_model", opts.activeModel);
    fd.append("sample_frames", String(opts.sampleFrames ?? 16));
    return jsonFetch<InspectionResult>("/api/inspect/video", { method: "POST", body: fd });
  },

  switchModel: (mode: "builtin" | "custom") => {
    const fd = new FormData();
    fd.append("mode", mode);
    return jsonFetch<{ active_model: "builtin" | "custom"; custom_classes: string[] }>(
      "/api/model/switch",
      { method: "POST", body: fd }
    );
  },

  trainStatus: () => jsonFetch<TrainingStatus>("/api/train/status"),

  startCustomTraining: async (opts: {
    items: Array<{ file: File; label: string }>;
    epochs: number;
    backbone: string;
    useDemoPack: boolean;
  }) => {
    const fd = new FormData();
    const labels: string[] = [];
    for (const item of opts.items) {
      fd.append("files", item.file);
      labels.push(item.label);
    }
    fd.append("labels_json", JSON.stringify(labels));
    fd.append("epochs", String(opts.epochs));
    fd.append("backbone", opts.backbone);
    fd.append("use_demo_custom_pack", String(opts.useDemoPack));
    return jsonFetch<{ started: boolean; class_names: string[]; sample_count: number }>(
      "/api/train/custom",
      { method: "POST", body: fd }
    );
  },
};
