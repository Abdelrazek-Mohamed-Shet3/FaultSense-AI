import React, { useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Download,
  Eye,
  FileJson,
  FileText,
  Film,
  Image as ImageIcon,
  Layers,
  Loader2,
  ShieldAlert,
  Sparkles,
  Upload,
} from "lucide-react";
import { api } from "../api";
import type { InspectionResult, Lang, SampleItem, SeverityLevel } from "../types";

interface Props {
  lang: Lang;
  activeModel: "builtin" | "custom";
  samples: { images: SampleItem[]; videos: SampleItem[] };
  result: InspectionResult | null;
  onResult: (res: InspectionResult) => void;
}

type ViewLayer = "blend" | "original" | "gradcam" | "localized";

function severityBadgeStyle(level: SeverityLevel) {
  if (level === "critical") {
    return "bg-terracotta-tint text-terracotta border-terracotta/25";
  }
  if (level === "warning") {
    return "bg-amberwarm-tint text-amberwarm border-amberwarm/25";
  }
  return "bg-sage-tint text-sage border-sage/25";
}

function severityIcon(level: SeverityLevel) {
  if (level === "critical") return <ShieldAlert className="h-4 w-4" />;
  if (level === "warning") return <AlertTriangle className="h-4 w-4" />;
  return <CheckCircle2 className="h-4 w-4" />;
}

export const InspectStudio: React.FC<Props> = ({
  lang,
  activeModel,
  samples,
  result,
  onResult,
}) => {
  const isAr = lang === "ar";
  const [mediaMode, setMediaMode] = useState<"image" | "video">("image");
  const [domainFilter, setDomainFilter] = useState<string>("all");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [viewLayer, setViewLayer] = useState<ViewLayer>("blend");
  const [blendOpacity, setBlendOpacity] = useState<number>(65);
  const [selectedKeyframeIdx, setSelectedKeyframeIdx] = useState<number>(0);
  const [dragActive, setDragActive] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const runFileInspection = async (file: File) => {
    setError(null);
    setLoading(true);
    setSelectedKeyframeIdx(0);
    try {
      const isVid = file.type.startsWith("video/") || /\.(mp4|mov|avi|webm|mkv)$/i.test(file.name);
      setMediaMode(isVid ? "video" : "image");
      const res = isVid
        ? await api.inspectVideo({ file, domainFilter, activeModel })
        : await api.inspectImage({ file, domainFilter, activeModel });
      onResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inspection failed");
    } finally {
      setLoading(false);
    }
  };

  const runSampleInspection = async (sample: SampleItem) => {
    setError(null);
    setLoading(true);
    setSelectedKeyframeIdx(0);
    setMediaMode(sample.media_type);
    try {
      const res =
        sample.media_type === "video"
          ? await api.inspectVideo({ sampleId: sample.id, domainFilter, activeModel })
          : await api.inspectImage({ sampleId: sample.id, domainFilter, activeModel });
      onResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sample inspection failed");
    } finally {
      setLoading(false);
    }
  };

  const activeKeyframe = useMemo(() => {
    if (!result || result.media_type !== "video" || !result.keyframes?.length) return null;
    return result.keyframes[Math.min(selectedKeyframeIdx, result.keyframes.length - 1)];
  }, [result, selectedKeyframeIdx]);

  const currentOrigUrl = activeKeyframe ? activeKeyframe.original_url : result?.assets.original;
  const currentCamUrl = activeKeyframe ? activeKeyframe.gradcam_url : result?.assets.gradcam;
  const currentLocUrl = activeKeyframe ? activeKeyframe.localized_url : result?.assets.localized;

  const domainOptions = [
    { id: "all", en: "All Domains (Auto)", ar: "جميع المجالات (تلقائي)" },
    { id: "industrial", en: "Industrial & PCB", ar: "الجودة الصناعية و PCB" },
    { id: "infrastructure", en: "Roads & Corrosion", ar: "الطرق والصدأ" },
    { id: "safety", en: "Safety & PPE", ar: "السلامة المهنية" },
  ];

  const displayedSamples = mediaMode === "image" ? samples.images : samples.videos;

  return (
    <div className="space-y-6">
      {/* Control Strip: Media Mode + Domain Filter */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-border bg-alabaster/70 p-4">
        <div className="flex items-center gap-2 rounded-xl border border-border bg-canvas p-1 shadow-soft">
          <button
            onClick={() => setMediaMode("image")}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
              mediaMode === "image"
                ? "bg-sage text-canvas shadow-sm"
                : "text-ink-muted hover:bg-stone hover:text-ink"
            }`}
          >
            <ImageIcon className="h-4 w-4" />
            <span>{isAr ? "فحص الصور" : "Image Inspection"}</span>
          </button>
          <button
            onClick={() => setMediaMode("video")}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
              mediaMode === "video"
                ? "bg-sage text-canvas shadow-sm"
                : "text-ink-muted hover:bg-stone hover:text-ink"
            }`}
          >
            <Film className="h-4 w-4" />
            <span>{isAr ? "فحص الفيديو" : "Video Temporal"}</span>
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-medium text-ink-subtle px-2">
            {isAr ? "نطاق التصنيف:" : "Domain Scope:"}
          </span>
          {domainOptions.map((opt) => (
            <button
              key={opt.id}
              disabled={activeModel === "custom"}
              onClick={() => setDomainFilter(opt.id)}
              className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition-all border ${
                domainFilter === opt.id
                  ? "border-sage/30 bg-sage-tint text-sage"
                  : "border-border bg-canvas text-ink-muted hover:bg-stone"
              } ${activeModel === "custom" ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {isAr ? opt.ar : opt.en}
            </button>
          ))}
        </div>
      </div>

      {/* Main Split Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Column: Uploader & Samples */}
        <div className="space-y-5 lg:col-span-5">
          {/* Drag & Drop Card */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              const f = e.dataTransfer.files?.[0];
              if (f) void runFileInspection(f);
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`group cursor-pointer rounded-2xl border-2 border-dashed p-7 text-center transition-all ${
              dragActive
                ? "border-sage bg-sage-tint/60"
                : "border-border bg-canvas hover:border-sage/50 hover:bg-alabaster"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={mediaMode === "video" ? "video/*" : "image/*,video/*"}
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void runFileInspection(f);
                e.target.value = "";
              }}
            />
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-sage-tint text-sage transition-transform group-hover:scale-105">
              {loading ? <Loader2 className="h-6 w-6 animate-spin" /> : <Upload className="h-6 w-6" />}
            </div>
            <h3 className="text-sm font-semibold text-ink">
              {loading
                ? isAr
                  ? "جاري التحليل بالشبكة العصبية العميقة..."
                  : "Running Deep Learning Inference..."
                : isAr
                ? "اسحب وأفلت صورة أو فيديو هنا، أو اضغط للاختيار"
                : "Drop an image or video here, or click to browse"}
            </h3>
            <p className="mt-1.5 text-xs text-ink-subtle">
              {isAr
                ? "يدعم PNG, JPG, WEBP و MP4, MOV, AVI مع خرائط Grad-CAM الفورية"
                : "Supports PNG, JPG, WEBP & MP4, MOV, AVI with instant Grad-CAM explainability"}
            </p>
          </div>

          {error && (
            <div className="rounded-xl border border-terracotta/30 bg-terracotta-tint p-3.5 text-xs text-terracotta">
              {error}
            </div>
          )}

          {/* Quick 1-Click Sample Gallery */}
          <div className="rounded-2xl border border-border bg-canvas p-4 shadow-soft">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-sand" />
                <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
                  {isAr ? "عينات جاهزة للتجربة الفورية" : "1-Click Benchmark Samples"}
                </h4>
              </div>
              <span className="rounded-full bg-stone px-2.5 py-0.5 text-[11px] font-medium text-ink-muted">
                {displayedSamples.length} {mediaMode === "image" ? (isAr ? "صورة" : "images") : isAr ? "فيديو" : "clips"}
              </span>
            </div>

            <div className="grid max-h-[340px] grid-cols-2 gap-2.5 overflow-y-auto pr-1">
              {displayedSamples.map((s) => (
                <button
                  key={s.id}
                  disabled={loading}
                  onClick={() => void runSampleInspection(s)}
                  className="group flex flex-col overflow-hidden rounded-xl border border-border bg-alabaster text-left transition-all hover:border-sage/40 hover:bg-canvas hover:shadow-soft disabled:opacity-50"
                >
                  <div className="relative h-24 w-full overflow-hidden bg-stone">
                    {s.media_type === "image" ? (
                      <img
                        src={s.url}
                        alt={s.label_en}
                        className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                      />
                    ) : (
                      <video
                        src={s.url}
                        muted
                        playsInline
                        preload="metadata"
                        className="h-full w-full object-cover"
                      />
                    )}
                    <span
                      className={`absolute top-2 right-2 rounded-full border px-2 py-0.5 text-[10px] font-semibold backdrop-blur-sm ${severityBadgeStyle(
                        s.severity_level
                      )}`}
                    >
                      {s.severity_level.toUpperCase()}
                    </span>
                  </div>
                  <div className="p-2.5">
                    <p className="line-clamp-1 text-xs font-semibold text-ink">
                      {isAr ? s.label_ar : s.label_en}
                    </p>
                    <p className="mt-0.5 text-[10px] uppercase tracking-wider text-ink-subtle">
                      {s.domain}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Visual Inspector & Diagnostics */}
        <div className="space-y-5 lg:col-span-7">
          {!result ? (
            <div className="flex min-h-[460px] flex-col items-center justify-center rounded-2xl border border-border bg-alabaster/50 p-10 text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-border bg-canvas text-sage shadow-soft">
                <Eye className="h-6 w-6" />
              </div>
              <h3 className="text-base font-semibold text-ink">
                {isAr ? "استوديو الفحص البصري الذكي" : "Visual Explainability & Fault Studio"}
              </h3>
              <p className="mt-2 max-w-md text-xs leading-relaxed text-ink-muted">
                {isAr
                  ? "ارفع أي صورة أو فيديو أو اختر عينة من القائمة لعرض التشخيص الفوري، خرائط الانتباه الحرارية Grad-CAM، تحديد مواضع الأخطاء، وتقارير PDF."
                  : "Upload an image or video, or pick any benchmark sample on the left to inspect spatial Grad-CAM heatmaps, localized fault regions, temporal timelines, and exportable PDF reports."}
              </p>
            </div>
          ) : (
            <>
              {/* Top Diagnostic Banner */}
              <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="space-y-1.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${severityBadgeStyle(
                          result.severity_level
                        )}`}
                      >
                        {severityIcon(result.severity_level)}
                        <span>{result.severity_level.toUpperCase()}</span>
                      </span>
                      <span className="rounded-full border border-border bg-alabaster px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide text-ink-muted">
                        {result.domain}
                      </span>
                      <span className="rounded-full bg-lavender-tint px-2.5 py-1 text-[11px] font-medium text-lavender">
                        {result.media_type === "video"
                          ? isAr
                            ? "تحليل فيديو زمني"
                            : "Temporal Video"
                          : isAr
                          ? "تحليل مكاني"
                          : "Spatial Image"}
                      </span>
                    </div>
                    <h2 className="text-xl font-bold text-ink">
                      {isAr ? result.label_ar : result.label_en}
                    </h2>
                    <p className="text-xs text-ink-muted">
                      {isAr ? result.description_ar : result.description_en}
                    </p>
                  </div>

                  {/* Export Buttons */}
                  <div className="flex items-center gap-2">
                    <a
                      href={`/api/reports/${result.id}/pdf`}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1.5 rounded-xl bg-sage px-3.5 py-2 text-xs font-semibold text-canvas shadow-sm transition-colors hover:bg-sage-hover"
                    >
                      <FileText className="h-3.5 w-3.5" />
                      <span>{isAr ? "تقرير PDF" : "PDF Report"}</span>
                    </a>
                    <a
                      href={`/api/reports/${result.id}/json`}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1.5 rounded-xl border border-border bg-alabaster px-3 py-2 text-xs font-semibold text-ink-muted transition-colors hover:bg-stone hover:text-ink"
                    >
                      <FileJson className="h-3.5 w-3.5" />
                      <span>JSON</span>
                    </a>
                  </div>
                </div>

                {/* KPI Mini Row */}
                <div className="mt-4 grid grid-cols-3 gap-3 border-t border-border pt-4">
                  <div className="rounded-xl bg-alabaster p-3">
                    <div className="text-[11px] font-medium text-ink-subtle">
                      {isAr ? "نسبة الثقة" : "Model Confidence"}
                    </div>
                    <div className="mt-1 font-mono text-lg font-bold text-ink">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl bg-alabaster p-3">
                    <div className="text-[11px] font-medium text-ink-subtle">
                      {isAr ? "مؤشر الخطورة" : "Severity Index"}
                    </div>
                    <div
                      className={`mt-1 font-mono text-lg font-bold ${
                        result.severity_level === "critical"
                          ? "text-terracotta"
                          : result.severity_level === "warning"
                          ? "text-amberwarm"
                          : "text-sage"
                      }`}
                    >
                      {result.severity_score} / 100
                    </div>
                  </div>
                  <div className="rounded-xl bg-alabaster p-3">
                    <div className="text-[11px] font-medium text-ink-subtle">
                      {isAr ? "المناطق المرصودة" : "Localized Regions"}
                    </div>
                    <div className="mt-1 font-mono text-lg font-bold text-ink">
                      {result.regions.length}
                    </div>
                  </div>
                </div>
              </div>

              {/* Video Annotated Player + Temporal Timeline (only for video) */}
              {result.media_type === "video" && result.assets.annotated_video && (
                <div className="rounded-2xl border border-border bg-canvas p-4 shadow-soft space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xs font-semibold text-ink">
                      <Film className="h-4 w-4 text-sage" />
                      <span>
                        {isAr ? "الفيديو المشروح بالتحليل الزمني" : "Annotated Temporal Output Video"}
                      </span>
                    </div>
                    {result.video_stats && (
                      <span className="font-mono text-[11px] text-ink-subtle">
                        {result.video_stats.duration_sec}s · {result.video_stats.total_frames}f ·{" "}
                        {isAr ? "ذروة الخطأ:" : "Peak:"} {result.video_stats.peak_timestamp_sec}s
                      </span>
                    )}
                  </div>

                  <video
                    key={result.assets.annotated_video}
                    src={result.assets.annotated_video}
                    controls
                    autoPlay
                    loop
                    playsInline
                    className="max-h-[300px] w-full rounded-xl border border-border bg-alabaster object-contain"
                  />

                  {/* Interactive Temporal Timeline Bar Chart */}
                  {result.timeline && result.timeline.length > 0 && (
                    <div className="rounded-xl border border-border bg-alabaster p-3">
                      <div className="mb-2 flex items-center justify-between text-[11px] font-medium text-ink-muted">
                        <span className="flex items-center gap-1.5">
                          <Activity className="h-3.5 w-3.5 text-lavender" />
                          {isAr
                            ? "الخط الزمني لاحتمالية الخطأ والانتباه الزمني"
                            : "Frame-by-Frame Fault Probability & Temporal Attention"}
                        </span>
                        <span className="text-ink-subtle">
                          {isAr ? "أعمدة التراكوتا = إطار خطأ" : "Terracotta = fault frame · Sage = clean"}
                        </span>
                      </div>
                      <div className="flex h-20 items-end gap-1 pt-2">
                        {result.timeline.map((pt) => {
                          const hPct = Math.max(12, Math.round(pt.target_class_prob * 100));
                          return (
                            <div
                              key={pt.sample_index}
                              title={`t=${pt.timestamp_sec}s | ${pt.label_en} (${Math.round(
                                pt.confidence * 100
                              )}%) | Attn: ${Math.round(pt.temporal_attention * 100)}%`}
                              className="group relative flex flex-1 flex-col items-center justify-end h-full"
                            >
                              <div
                                style={{ height: `${hPct}%` }}
                                className={`w-full rounded-t-md transition-all ${
                                  pt.is_fault
                                    ? "bg-terracotta/80 group-hover:bg-terracotta"
                                    : "bg-sage/70 group-hover:bg-sage"
                                }`}
                              />
                              <span className="mt-1 font-mono text-[9px] text-ink-subtle">
                                {pt.timestamp_sec.toFixed(1)}s
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Peak Keyframe Selector */}
                  {result.keyframes && result.keyframes.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[11px] font-medium text-ink-subtle">
                        {isAr ? "أهم الإطارات (Grad-CAM):" : "Peak Keyframes:"}
                      </span>
                      {result.keyframes.map((kf, idx) => (
                        <button
                          key={kf.rank}
                          onClick={() => setSelectedKeyframeIdx(idx)}
                          className={`rounded-lg border px-2.5 py-1 text-xs font-medium transition-all ${
                            selectedKeyframeIdx === idx
                              ? "border-sage bg-sage-tint text-sage"
                              : "border-border bg-alabaster text-ink-muted hover:bg-stone"
                          }`}
                        >
                          #{kf.rank} · {kf.timestamp_sec}s ({Math.round(kf.attention_weight * 100)}% attn)
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Grad-CAM Visual Comparison Panel */}
              <div className="rounded-2xl border border-border bg-canvas p-4 shadow-soft space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-xs font-semibold text-ink">
                    <Layers className="h-4 w-4 text-sage" />
                    <span>
                      {isAr ? "تفسير الشبكة العصبية (Grad-CAM)" : "Neural Explainability & Localization"}
                    </span>
                  </div>

                  <div className="flex flex-wrap items-center gap-1 rounded-xl border border-border bg-alabaster p-1">
                    {(
                      [
                        { id: "blend", en: "Blend", ar: "دمج تفاعلي" },
                        { id: "original", en: "Original", ar: "الأصلية" },
                        { id: "gradcam", en: "Grad-CAM", ar: "خريطة Grad-CAM" },
                        { id: "localized", en: "Regions", ar: "تحديد المناطق" },
                      ] as const
                    ).map((tab) => (
                      <button
                        key={tab.id}
                        onClick={() => setViewLayer(tab.id)}
                        className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
                          viewLayer === tab.id
                            ? "bg-canvas text-ink shadow-sm"
                            : "text-ink-muted hover:text-ink"
                        }`}
                      >
                        {isAr ? tab.ar : tab.en}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Canvas Display */}
                <div className="relative flex min-h-[260px] items-center justify-center overflow-hidden rounded-xl border border-border bg-alabaster p-3">
                  {viewLayer === "blend" && currentOrigUrl && currentCamUrl ? (
                    <div className="relative inline-block overflow-hidden rounded-lg">
                      <img
                        src={currentOrigUrl}
                        alt="Original"
                        className="max-h-[320px] w-auto rounded-lg object-contain"
                      />
                      <img
                        src={currentCamUrl}
                        alt="Grad-CAM Overlay"
                        style={{ opacity: blendOpacity / 100 }}
                        className="absolute inset-0 h-full w-full rounded-lg object-contain transition-opacity"
                      />
                    </div>
                  ) : (
                    <img
                      src={
                        viewLayer === "original"
                          ? currentOrigUrl
                          : viewLayer === "gradcam"
                          ? currentCamUrl
                          : currentLocUrl
                      }
                      alt={viewLayer}
                      className="max-h-[320px] w-auto rounded-lg object-contain"
                    />
                  )}
                </div>

                {viewLayer === "blend" && (
                  <div className="flex items-center gap-3 px-1">
                    <span className="text-[11px] font-medium text-ink-muted">
                      {isAr ? "شفافية خريطة الحرارة:" : "Heatmap Opacity:"}
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={blendOpacity}
                      onChange={(e) => setBlendOpacity(Number(e.target.value))}
                      className="flex-1 cursor-pointer"
                    />
                    <span className="font-mono text-xs text-ink-muted w-9 text-right">
                      {blendOpacity}%
                    </span>
                  </div>
                )}

                {/* Bottom Grid: Probabilities + Recommendation */}
                <div className="grid grid-cols-1 gap-4 pt-2 md:grid-cols-2">
                  <div className="rounded-xl border border-border bg-alabaster p-3.5">
                    <h4 className="mb-2.5 text-xs font-semibold text-ink">
                      {isAr ? "توزيع احتمالات التصنيف" : "Top Class Probabilities"}
                    </h4>
                    <div className="space-y-2">
                      {result.probabilities.slice(0, 4).map((p) => (
                        <div key={p.class_id} className="space-y-1">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="font-medium text-ink truncate max-w-[190px]">
                              {isAr ? p.label_ar : p.label_en}
                            </span>
                            <span className="font-mono text-ink-muted">
                              {(p.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-1.5 w-full overflow-hidden rounded-full bg-stone">
                            <div
                              style={{
                                width: `${Math.max(2, p.probability * 100)}%`,
                                backgroundColor: p.color,
                              }}
                              className="h-full rounded-full transition-all"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex flex-col justify-between rounded-xl border border-border bg-sage-tint/50 p-3.5">
                    <div>
                      <h4 className="mb-1.5 text-xs font-semibold text-sage">
                        {isAr ? "التوصية الهندسية المقترحة" : "Recommended Action"}
                      </h4>
                      <p className="text-xs leading-relaxed text-ink">
                        {isAr ? result.recommendation_ar : result.recommendation_en}
                      </p>
                    </div>
                    <div className="mt-3 flex items-center justify-between border-t border-sage/15 pt-2.5 text-[11px] text-ink-muted">
                      <span>ID: {result.id}</span>
                      <a
                        href={`/api/reports/${result.id}/pdf`}
                        className="inline-flex items-center gap-1 font-semibold text-sage hover:underline"
                      >
                        <Download className="h-3 w-3" />
                        {isAr ? "تحميل التقرير" : "Export PDF"}
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
