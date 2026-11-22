import React from "react";
import { Eye, FileText, Film, Image as ImageIcon, Trash2 } from "lucide-react";
import type { InspectionResult, Lang } from "../types";

interface Props {
  lang: Lang;
  items: InspectionResult[];
  onInspectSelect: (item: InspectionResult) => void;
  onDelete: (id: string) => void;
}

export const HistoryView: React.FC<Props> = ({ lang, items, onInspectSelect, onDelete }) => {
  const isAr = lang === "ar";

  if (!items.length) {
    return (
      <div className="rounded-2xl border border-border bg-alabaster p-12 text-center text-sm text-ink-muted">
        {isAr
          ? "لا توجد عمليات فحص مسجلة بعد. ابدأ بفحص صورة أو فيديو من الاستوديو!"
          : "No inspection records yet. Analyze an image or video in Inspect Studio first!"}
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-border bg-canvas shadow-soft overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-border bg-alabaster text-[11px] uppercase tracking-wider text-ink-subtle">
            <tr>
              <th className="px-4 py-3 font-medium">{isAr ? "المعاينة" : "Preview"}</th>
              <th className="px-4 py-3 font-medium">{isAr ? "التصنيف" : "Classification"}</th>
              <th className="px-4 py-3 font-medium">{isAr ? "النوع / المجال" : "Media / Domain"}</th>
              <th className="px-4 py-3 font-medium">{isAr ? "الثقة والخطورة" : "Confidence / Severity"}</th>
              <th className="px-4 py-3 font-medium">{isAr ? "التاريخ" : "Timestamp"}</th>
              <th className="px-4 py-3 font-medium text-right">{isAr ? "إجراءات" : "Actions"}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {items.map((item) => (
              <tr key={item.id} className="hover:bg-alabaster/70 transition-colors">
                <td className="px-4 py-3">
                  <img
                    src={item.assets.localized || item.assets.original}
                    alt={item.label_en}
                    className="h-12 w-12 rounded-lg border border-border object-cover"
                  />
                </td>
                <td className="px-4 py-3">
                  <div className="font-semibold text-ink">
                    {isAr ? item.label_ar : item.label_en}
                  </div>
                  <div className="font-mono text-[11px] text-ink-subtle">{item.filename || item.id}</div>
                </td>
                <td className="px-4 py-3">
                  <div className="inline-flex items-center gap-1 rounded-full border border-border bg-alabaster px-2.5 py-0.5 text-[11px] text-ink-muted">
                    {item.media_type === "video" ? (
                      <Film className="h-3 w-3 text-lavender" />
                    ) : (
                      <ImageIcon className="h-3 w-3 text-sage" />
                    )}
                    <span className="uppercase">{item.domain}</span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold text-ink">
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                        item.severity_level === "critical"
                          ? "bg-terracotta-tint text-terracotta"
                          : item.severity_level === "warning"
                          ? "bg-amberwarm-tint text-amberwarm"
                          : "bg-sage-tint text-sage"
                      }`}
                    >
                      {item.severity_score}/100
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 font-mono text-[11px] text-ink-subtle">
                  {item.created_at}
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="inline-flex items-center gap-1.5">
                    <button
                      onClick={() => onInspectSelect(item)}
                      title={isAr ? "عرض في الاستوديو" : "Open in Studio"}
                      className="rounded-lg border border-border bg-canvas p-1.5 text-ink-muted hover:bg-sage-tint hover:text-sage"
                    >
                      <Eye className="h-3.5 w-3.5" />
                    </button>
                    <a
                      href={`/api/reports/${item.id}/pdf`}
                      target="_blank"
                      rel="noreferrer"
                      title="PDF Report"
                      className="rounded-lg border border-border bg-canvas p-1.5 text-ink-muted hover:bg-sage-tint hover:text-sage"
                    >
                      <FileText className="h-3.5 w-3.5" />
                    </a>
                    <button
                      onClick={() => onDelete(item.id)}
                      title={isAr ? "حذف" : "Delete"}
                      className="rounded-lg border border-border bg-canvas p-1.5 text-ink-subtle hover:bg-terracotta-tint hover:text-terracotta"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
