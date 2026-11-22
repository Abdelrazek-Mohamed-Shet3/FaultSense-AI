import React, { useCallback, useEffect, useState } from "react";
import {
  BarChart3,
  Cpu,
  History,
  Languages,
  ScanSearch,
  Sliders,
  Sparkles,
} from "lucide-react";
import { api } from "./api";
import { AnalyticsView } from "./components/AnalyticsView";
import { CustomTrainerView } from "./components/CustomTrainerView";
import { HistoryView } from "./components/HistoryView";
import { InspectStudio } from "./components/InspectStudio";
import type {
  HealthInfo,
  InspectionResult,
  Lang,
  ModelMetrics,
  SampleItem,
} from "./types";

type TabId = "inspect" | "analytics" | "trainer" | "history";

export default function App() {
  const [lang, setLang] = useState<Lang>("en");
  const [tab, setTab] = useState<TabId>("inspect");
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [samples, setSamples] = useState<{ images: SampleItem[]; videos: SampleItem[] }>({
    images: [],
    videos: [],
  });
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [historyItems, setHistoryItems] = useState<InspectionResult[]>([]);
  const [currentResult, setCurrentResult] = useState<InspectionResult | null>(null);

  const isAr = lang === "ar";

  const refreshAll = useCallback(async () => {
    try {
      const [h, s, m, hist] = await Promise.all([
        api.health(),
        api.samples(),
        api.metrics(),
        api.history(),
      ]);
      setHealth(h);
      setSamples(s);
      setMetrics(m);
      setHistoryItems(hist.items);
    } catch (err) {
      console.error("Failed to load initial data", err);
    }
  }, []);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  const handleModelSwitch = async (mode: "builtin" | "custom") => {
    try {
      await api.switchModel(mode);
      const h = await api.health();
      setHealth(h);
    } catch (err) {
      console.error(err);
    }
  };

  const handleNewResult = (res: InspectionResult) => {
    setCurrentResult(res);
    setHistoryItems((prev) => [res, ...prev.filter((item) => item.id !== res.id)]);
  };

  const handleDeleteHistory = async (id: string) => {
    try {
      await api.deleteHistory(id);
      setHistoryItems((prev) => prev.filter((item) => item.id !== id));
    } catch (err) {
      console.error(err);
    }
  };

  const navItems: Array<{ id: TabId; en: string; ar: string; icon: React.ReactNode }> = [
    {
      id: "inspect",
      en: "Inspect Studio",
      ar: "استوديو الفحص",
      icon: <ScanSearch className="h-4 w-4" />,
    },
    {
      id: "analytics",
      en: "Model Analytics",
      ar: "أداء النموذج",
      icon: <BarChart3 className="h-4 w-4" />,
    },
    {
      id: "trainer",
      en: "Custom Trainer",
      ar: "التدريب المخصص",
      icon: <Sliders className="h-4 w-4" />,
    },
    {
      id: "history",
      en: "History Log",
      ar: "سجل الفحوصات",
      icon: <History className="h-4 w-4" />,
    },
  ];

  return (
    <div dir={isAr ? "rtl" : "ltr"} className="min-h-screen bg-canvas text-ink">
      {/* Pure White Sticky Topbar */}
      <header className="sticky top-0 z-30 border-b border-border bg-canvas/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-3.5">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-sage/20 bg-sage-tint text-sage">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-ink">FaultSense AI</h1>
                <span className="rounded-full border border-border bg-alabaster px-2 py-0.5 font-mono text-[10px] font-medium text-ink-muted">
                  v1.0
                </span>
              </div>
              <p className="text-[11px] text-ink-subtle">
                {isAr
                  ? "منصة التعلم العميق متعددة المجالات لتصنيف أخطاء الصور والفيديو"
                  : "Multi-Domain Spatio-Temporal Deep Learning Inspection"}
              </p>
            </div>
          </div>

          {/* Center Navigation Pills */}
          <nav className="flex flex-wrap items-center gap-1 rounded-xl border border-border bg-alabaster p-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                className={`flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-all ${
                  tab === item.id
                    ? "bg-canvas text-sage shadow-soft border border-border/60"
                    : "text-ink-muted hover:text-ink"
                }`}
              >
                {item.icon}
                <span>{isAr ? item.ar : item.en}</span>
                {item.id === "history" && historyItems.length > 0 && (
                  <span className="rounded-full bg-sage-tint px-1.5 py-0.2 font-mono text-[10px] text-sage">
                    {historyItems.length}
                  </span>
                )}
              </button>
            ))}
          </nav>

          {/* Right Status Controls */}
          <div className="flex items-center gap-2">
            {health?.has_custom_model && (
              <div className="flex items-center rounded-xl border border-border bg-alabaster p-0.5 text-[11px] font-medium">
                <button
                  onClick={() => void handleModelSwitch("builtin")}
                  className={`rounded-lg px-2.5 py-1 transition-all ${
                    health.active_model === "builtin"
                      ? "bg-canvas text-sage shadow-sm"
                      : "text-ink-muted"
                  }`}
                >
                  {isAr ? "الأساسي" : "Built-in"}
                </button>
                <button
                  onClick={() => void handleModelSwitch("custom")}
                  className={`rounded-lg px-2.5 py-1 transition-all ${
                    health.active_model === "custom"
                      ? "bg-canvas text-lavender shadow-sm"
                      : "text-ink-muted"
                  }`}
                >
                  {isAr ? "المخصص" : "Custom"}
                </button>
              </div>
            )}

            <div className="hidden sm:inline-flex items-center gap-1.5 rounded-xl border border-border bg-alabaster px-2.5 py-1.5 font-mono text-[11px] text-ink-muted">
              <Cpu className="h-3.5 w-3.5 text-sage" />
              <span className="uppercase">{health?.device || "mps"}</span>
            </div>

            <button
              onClick={() => setLang((prev) => (prev === "en" ? "ar" : "en"))}
              className="inline-flex items-center gap-1.5 rounded-xl border border-border bg-canvas px-3 py-1.5 text-xs font-semibold text-ink-muted transition-colors hover:bg-stone hover:text-ink"
            >
              <Languages className="h-3.5 w-3.5 text-sand" />
              <span>{isAr ? "English" : "العربية"}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="mx-auto max-w-7xl px-5 py-7">
        {tab === "inspect" && (
          <InspectStudio
            lang={lang}
            activeModel={health?.active_model ?? "builtin"}
            samples={samples}
            result={currentResult}
            onResult={handleNewResult}
          />
        )}
        {tab === "analytics" && <AnalyticsView lang={lang} metrics={metrics} />}
        {tab === "trainer" && <CustomTrainerView lang={lang} onModelTrained={refreshAll} />}
        {tab === "history" && (
          <HistoryView
            lang={lang}
            items={historyItems}
            onInspectSelect={(item) => {
              setCurrentResult(item);
              setTab("inspect");
            }}
            onDelete={(id) => void handleDeleteHistory(id)}
          />
        )}
      </main>
    </div>
  );
}
