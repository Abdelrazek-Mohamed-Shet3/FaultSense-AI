import React from "react";
import { Award, Cpu, Layers, TrendingUp } from "lucide-react";
import type { Lang, ModelMetrics } from "../types";

interface Props {
  lang: Lang;
  metrics: ModelMetrics | null;
}

export const AnalyticsView: React.FC<Props> = ({ lang, metrics }) => {
  const isAr = lang === "ar";

  if (!metrics || !metrics.available) {
    return (
      <div className="rounded-2xl border border-border bg-alabaster p-10 text-center text-sm text-ink-muted">
        {isAr ? "جاري تحميل مقاييس أداء النموذج..." : "Loading model benchmark metrics..."}
      </div>
    );
  }

  const hist = metrics.spatial_history;
  const maxLoss = Math.max(1.2, ...hist.map((h) => Math.max(h.train_loss, h.val_loss)));

  const buildPolyline = (values: number[], maxVal: number) => {
    if (values.length <= 1) return "";
    return values
      .map((v, i) => {
        const x = (i / (values.length - 1)) * 280 + 20;
        const y = 120 - (v / maxVal) * 100 + 10;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const trainLossPts = buildPolyline(
    hist.map((h) => h.train_loss),
    maxLoss
  );
  const valLossPts = buildPolyline(
    hist.map((h) => h.val_loss),
    maxLoss
  );
  const trainAccPts = buildPolyline(
    hist.map((h) => h.train_acc),
    1.05
  );
  const valAccPts = buildPolyline(
    hist.map((h) => h.val_acc),
    1.05
  );

  return (
    <div className="space-y-6">
      {/* Top KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft">
          <div className="flex items-center justify-between text-xs font-medium text-ink-subtle">
            <span>{isAr ? "دقة الاختبار (صور)" : "Test Accuracy (Image)"}</span>
            <Award className="h-4 w-4 text-sage" />
          </div>
          <div className="mt-2 font-mono text-2xl font-bold text-sage">
            {(metrics.test_accuracy * 100).toFixed(1)}%
          </div>
          <div className="mt-1 text-[11px] text-ink-muted">
            {metrics.test_samples} {isAr ? "عينة اختبار مستقلة" : "held-out test samples"}
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft">
          <div className="flex items-center justify-between text-xs font-medium text-ink-subtle">
            <span>{isAr ? "معامل F1 الشامل" : "Macro F1-Score"}</span>
            <TrendingUp className="h-4 w-4 text-sand" />
          </div>
          <div className="mt-2 font-mono text-2xl font-bold text-ink">
            {(metrics.test_f1_macro * 100).toFixed(1)}%
          </div>
          <div className="mt-1 text-[11px] text-ink-muted">
            Precision {(metrics.test_precision_macro * 100).toFixed(1)}% · Recall{" "}
            {(metrics.test_recall_macro * 100).toFixed(1)}%
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft">
          <div className="flex items-center justify-between text-xs font-medium text-ink-subtle">
            <span>{isAr ? "دقة الفيديو الزمنية" : "Video Temporal Acc"}</span>
            <Layers className="h-4 w-4 text-lavender" />
          </div>
          <div className="mt-2 font-mono text-2xl font-bold text-lavender">
            {(metrics.video_val_accuracy * 100).toFixed(1)}%
          </div>
          <div className="mt-1 text-[11px] text-ink-muted">
            {isAr ? "محول انتباه زمني 12-إطار" : "12-frame Transformer Attention"}
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft">
          <div className="flex items-center justify-between text-xs font-medium text-ink-subtle">
            <span>{isAr ? "المعمارية والعتاد" : "Backbone & Device"}</span>
            <Cpu className="h-4 w-4 text-ink-muted" />
          </div>
          <div className="mt-2 font-mono text-lg font-bold text-ink uppercase">
            {metrics.device}
          </div>
          <div className="mt-1 text-[11px] text-ink-muted">
            {metrics.backbone} + SE · {metrics.training_seconds}s
          </div>
        </div>
      </div>

      {/* Curves Row */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {/* Loss Curve */}
        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-ink">
              {isAr ? "منحنى الخسارة (التدريب والتحقق)" : "Training & Validation Loss"}
            </h3>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1.5 text-sage font-medium">
                <span className="h-2 w-2 rounded-full bg-sage" /> Train
              </span>
              <span className="flex items-center gap-1.5 text-terracotta font-medium">
                <span className="h-2 w-2 rounded-full bg-terracotta" /> Val
              </span>
            </div>
          </div>
          <svg viewBox="0 0 320 140" className="w-full h-44 rounded-xl bg-alabaster border border-border">
            {[35, 70, 105].map((y) => (
              <line key={y} x1="20" y1={y} x2="300" y2={y} stroke="#E8E4DA" strokeDasharray="3 3" />
            ))}
            <polyline fill="none" stroke="#4E7A65" strokeWidth="2.5" strokeLinecap="round" points={trainLossPts} />
            <polyline fill="none" stroke="#B85D4E" strokeWidth="2.5" strokeLinecap="round" points={valLossPts} />
          </svg>
        </div>

        {/* Accuracy Curve */}
        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-ink">
              {isAr ? "منحنى الدقة عبر الحقب" : "Epoch Accuracy Convergence"}
            </h3>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1.5 text-sage font-medium">
                <span className="h-2 w-2 rounded-full bg-sage" /> Train Acc
              </span>
              <span className="flex items-center gap-1.5 text-sand font-medium">
                <span className="h-2 w-2 rounded-full bg-sand" /> Val Acc
              </span>
            </div>
          </div>
          <svg viewBox="0 0 320 140" className="w-full h-44 rounded-xl bg-alabaster border border-border">
            {[35, 70, 105].map((y) => (
              <line key={y} x1="20" y1={y} x2="300" y2={y} stroke="#E8E4DA" strokeDasharray="3 3" />
            ))}
            <polyline fill="none" stroke="#4E7A65" strokeWidth="2.5" strokeLinecap="round" points={trainAccPts} />
            <polyline fill="none" stroke="#B8886E" strokeWidth="2.5" strokeLinecap="round" points={valAccPts} />
          </svg>
        </div>
      </div>

      {/* Confusion Matrix + Per-Class Table */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft lg:col-span-6 overflow-x-auto">
          <h3 className="mb-3 text-sm font-semibold text-ink">
            {isAr ? "مصفوفة الارتباك (10 فئات)" : "Multi-Domain Confusion Matrix (10 Classes)"}
          </h3>
          <div className="inline-block min-w-full">
            <table className="w-full border-collapse text-center font-mono text-[11px]">
              <thead>
                <tr>
                  <th className="p-1.5 text-left text-ink-subtle">True \ Pred</th>
                  {metrics.class_ids.map((cid, idx) => (
                    <th key={cid} title={cid} className="p-1.5 text-ink-muted">
                      C{idx}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.confusion_matrix.map((row, rIdx) => {
                  const rowMax = Math.max(1, ...row);
                  return (
                    <tr key={metrics.class_ids[rIdx]} className="border-t border-border">
                      <td className="p-1.5 text-left font-sans text-[11px] font-medium text-ink">
                        C{rIdx} · {metrics.class_ids[rIdx].split("_").slice(1).join("_")}
                      </td>
                      {row.map((val, cIdx) => {
                        const intensity = val / rowMax;
                        const bg =
                          val === 0
                            ? "bg-alabaster text-ink-subtle"
                            : rIdx === cIdx
                            ? "bg-sage text-canvas font-bold"
                            : "bg-terracotta-tint text-terracotta font-semibold";
                        return (
                          <td
                            key={cIdx}
                            style={rIdx === cIdx && val > 0 ? { opacity: 0.45 + 0.55 * intensity } : undefined}
                            className={`p-1.5 rounded-sm ${bg}`}
                          >
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-canvas p-5 shadow-soft lg:col-span-6 overflow-x-auto">
          <h3 className="mb-3 text-sm font-semibold text-ink">
            {isAr ? "أداء التصنيف لكل فئة خطأ" : "Per-Class Precision, Recall & F1"}
          </h3>
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-border text-[11px] text-ink-subtle">
                <th className="pb-2 font-medium">{isAr ? "الفئة" : "Class"}</th>
                <th className="pb-2 font-medium">{isAr ? "المجال" : "Domain"}</th>
                <th className="pb-2 font-mono font-medium">P</th>
                <th className="pb-2 font-mono font-medium">R</th>
                <th className="pb-2 font-mono font-medium">F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {metrics.per_class.map((c) => (
                <tr key={c.id} className="hover:bg-alabaster">
                  <td className="py-2 pr-2 font-medium text-ink">
                    {isAr ? c.label_ar : c.label_en}
                  </td>
                  <td className="py-2 pr-2 text-[11px] uppercase text-ink-subtle">{c.domain}</td>
                  <td className="py-2 font-mono text-ink-muted">{(c.precision * 100).toFixed(0)}%</td>
                  <td className="py-2 font-mono text-ink-muted">{(c.recall * 100).toFixed(0)}%</td>
                  <td className="py-2 font-mono font-semibold text-sage">
                    {(c.f1 * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
