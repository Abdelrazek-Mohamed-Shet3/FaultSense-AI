import React, { useEffect, useState } from "react";
import { CheckCircle2, FolderPlus, Loader2, Play, Plus, Sparkles, Trash2 } from "lucide-react";
import { api } from "../api";
import type { Lang, TrainingStatus } from "../types";

interface Props {
  lang: Lang;
  onModelTrained: () => Promise<void>;
}

interface ClassBucket {
  name: string;
  files: File[];
}

export const CustomTrainerView: React.FC<Props> = ({ lang, onModelTrained }) => {
  const isAr = lang === "ar";
  const [buckets, setBuckets] = useState<ClassBucket[]>([
    { name: "normal_part", files: [] },
    { name: "defect_fault", files: [] },
  ]);
  const [newClassName, setNewClassName] = useState("");
  const [epochs, setEpochs] = useState(6);
  const [backbone, setBackbone] = useState("mobilenet_v3_small");
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollStatus = async () => {
    try {
      const st = await api.trainStatus();
      setStatus(st);
      if (st.state === "completed") {
        await onModelTrained();
      }
      return st;
    } catch {
      return null;
    }
  };

  useEffect(() => {
    void pollStatus();
  }, []);

  useEffect(() => {
    if (status?.state !== "running") return;
    const timer = setInterval(() => {
      void pollStatus();
    }, 1000);
    return () => clearInterval(timer);
  }, [status?.state]);

  const addClass = () => {
    const clean = newClassName.trim().replace(/\s+/g, "_");
    if (!clean || buckets.some((b) => b.name === clean)) return;
    setBuckets((prev) => [...prev, { name: clean, files: [] }]);
    setNewClassName("");
  };

  const launchTraining = async (useDemoPack: boolean) => {
    setError(null);
    const items: Array<{ file: File; label: string }> = [];
    if (!useDemoPack) {
      for (const b of buckets) {
        for (const f of b.files) items.push({ file: f, label: b.name });
      }
      if (items.length < 4) {
        setError(
          isAr
            ? "يرجى رفع 4 صور على الأقل موزعة على فئتين، أو استخدام حزمة التدريب التجريبية."
            : "Upload at least 4 images across 2+ classes, or click Quick Demo Fine-Tune."
        );
        return;
      }
    }

    try {
      await api.startCustomTraining({ items, epochs, backbone, useDemoPack });
      await pollStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
    }
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Left: Dataset Builder */}
      <div className="space-y-5 lg:col-span-7">
        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-bold text-ink">
                {isAr ? "استوديو التدريب المخصص (Custom Domain)" : "Universal Custom Domain Trainer"}
              </h2>
              <p className="mt-1 text-xs text-ink-muted">
                {isAr
                  ? "أنشئ فئات الأخطاء الخاصة بك وارفع صورًا لتدريب شبكة عصبية مخصصة فورًا على جهازك."
                  : "Define custom fault classes, upload labeled images, and fine-tune a live PyTorch model on MPS/CPU."}
              </p>
            </div>
            <button
              disabled={status?.state === "running"}
              onClick={() => void launchTraining(true)}
              className="inline-flex items-center gap-1.5 rounded-xl border border-sand/40 bg-sand-tint px-3.5 py-2 text-xs font-semibold text-ink transition-colors hover:bg-sand/20 disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5 text-sand" />
              <span>{isAr ? "تدريب تجريبي فوري" : "Quick Demo Fine-Tune"}</span>
            </button>
          </div>

          {/* Add Class Row */}
          <div className="flex gap-2">
            <input
              type="text"
              value={newClassName}
              onChange={(e) => setNewClassName(e.target.value)}
              placeholder={isAr ? "اسم فئة جديدة (مثل: missing_screw)" : "New class name (e.g. missing_screw)"}
              className="flex-1 rounded-xl border border-border bg-alabaster px-3.5 py-2 text-xs text-ink placeholder:text-ink-subtle"
            />
            <button
              onClick={addClass}
              className="inline-flex items-center gap-1.5 rounded-xl border border-border bg-canvas px-4 py-2 text-xs font-semibold text-ink hover:bg-stone"
            >
              <Plus className="h-3.5 w-3.5" />
              <span>{isAr ? "إضافة فئة" : "Add Class"}</span>
            </button>
          </div>

          {/* Class Buckets */}
          <div className="space-y-3">
            {buckets.map((bucket, idx) => (
              <div
                key={bucket.name}
                className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-alabaster p-3.5"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <FolderPlus className="h-4 w-4 text-sage" />
                    <span className="font-mono text-xs font-semibold text-ink">{bucket.name}</span>
                    <span className="rounded-full bg-canvas px-2 py-0.5 text-[11px] font-medium text-ink-muted border border-border">
                      {bucket.files.length} {isAr ? "ملف" : "files"}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <label className="cursor-pointer rounded-lg border border-sage/30 bg-sage-tint px-3 py-1.5 text-xs font-medium text-sage hover:bg-sage/15">
                    {isAr ? "رفع صور" : "Select Images"}
                    <input
                      type="file"
                      multiple
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => {
                        const selected = Array.from(e.target.files ?? []);
                        setBuckets((prev) =>
                          prev.map((b, i) => (i === idx ? { ...b, files: [...b.files, ...selected] } : b))
                        );
                      }}
                    />
                  </label>
                  {buckets.length > 2 && (
                    <button
                      onClick={() => setBuckets((prev) => prev.filter((_, i) => i !== idx))}
                      className="rounded-lg p-1.5 text-ink-subtle hover:bg-terracotta-tint hover:text-terracotta"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Hyperparameters */}
          <div className="grid grid-cols-2 gap-3 border-t border-border pt-4">
            <div>
              <label className="block text-[11px] font-medium text-ink-muted mb-1">
                {isAr ? "عدد الحقب (Epochs)" : "Training Epochs"}
              </label>
              <input
                type="number"
                min={2}
                max={25}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full rounded-xl border border-border bg-alabaster px-3 py-2 font-mono text-xs text-ink"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-ink-muted mb-1">
                {isAr ? "الشبكة الأساسية (Backbone)" : "Spatial Backbone"}
              </label>
              <select
                value={backbone}
                onChange={(e) => setBackbone(e.target.value)}
                className="w-full rounded-xl border border-border bg-alabaster px-3 py-2 text-xs text-ink"
              >
                <option value="mobilenet_v3_small">MobileNetV3-Small + SE (Fast)</option>
                <option value="resnet18">ResNet-18 + SE (Deep)</option>
              </select>
            </div>
          </div>

          {error && (
            <div className="rounded-xl border border-terracotta/30 bg-terracotta-tint p-3 text-xs text-terracotta">
              {error}
            </div>
          )}

          <button
            disabled={status?.state === "running"}
            onClick={() => void launchTraining(false)}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-sage py-2.5 text-xs font-semibold text-canvas shadow-sm transition-colors hover:bg-sage-hover disabled:opacity-50"
          >
            {status?.state === "running" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            <span>
              {status?.state === "running"
                ? isAr
                  ? "جاري التدريب..."
                  : "Training in Progress..."
                : isAr
                ? "بدء التدريب المخصص وتفعيل النموذج"
                : "Train Custom Model & Hot-Swap"}
            </span>
          </button>
        </div>
      </div>

      {/* Right: Live Training Monitor */}
      <div className="lg:col-span-5">
        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-ink">
              {isAr ? "مراقبة التدريب المباشر" : "Live Training Telemetry"}
            </h3>
            <span
              className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                status?.state === "running"
                  ? "bg-amberwarm-tint text-amberwarm"
                  : status?.state === "completed"
                  ? "bg-sage-tint text-sage"
                  : status?.state === "error"
                  ? "bg-terracotta-tint text-terracotta"
                  : "bg-stone text-ink-muted"
              }`}
            >
              {status?.state?.toUpperCase() ?? "IDLE"}
            </span>
          </div>

          {status && status.total_epochs > 0 && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-ink-muted">
                <span>
                  Epoch {status.epoch} / {status.total_epochs}
                </span>
                <span className="font-mono font-semibold text-sage">
                  Best Val Acc: {(status.best_val_acc * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-stone">
                <div
                  style={{
                    width: `${Math.round((status.epoch / Math.max(1, status.total_epochs)) * 100)}%`,
                  }}
                  className="h-full rounded-full bg-sage transition-all"
                />
              </div>
            </div>
          )}

          {status?.state === "completed" && (
            <div className="flex items-center gap-2 rounded-xl border border-sage/25 bg-sage-tint p-3 text-xs text-sage">
              <CheckCircle2 className="h-4 w-4 shrink-0" />
              <span>
                {isAr
                  ? "تم تدريب النموذج المخصص وتفعيله تلقائيًا في استوديو الفحص!"
                  : "Custom model trained & hot-swapped into Inspect Studio!"}
              </span>
            </div>
          )}

          {status?.history && status.history.length > 0 ? (
            <div className="max-h-72 overflow-y-auto rounded-xl border border-border bg-alabaster p-3">
              <table className="w-full text-left font-mono text-[11px]">
                <thead>
                  <tr className="border-b border-border text-ink-subtle">
                    <th className="pb-1.5">Epoch</th>
                    <th className="pb-1.5">Loss</th>
                    <th className="pb-1.5">Train</th>
                    <th className="pb-1.5">Val</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {status.history.map((h) => (
                    <tr key={h.epoch}>
                      <td className="py-1.5">{h.epoch}</td>
                      <td className="py-1.5">{h.train_loss.toFixed(4)}</td>
                      <td className="py-1.5">{(h.train_acc * 100).toFixed(1)}%</td>
                      <td className="py-1.5 font-semibold text-sage">
                        {(h.val_acc * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-border bg-alabaster p-8 text-center text-xs text-ink-subtle">
              {isAr
                ? "ستظهر منحنيات التدريب والدقة المباشرة هنا فور بدء التدريب."
                : "Epoch loss and validation accuracy logs will appear here once training starts."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
