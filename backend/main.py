"""FaultSense-DL FastAPI Application Server."""

import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.database import InspectionDB
from ml.dataset import load_custom_image_folder, materialize_dataset_and_samples
from ml.inference import FaultSenseEngine
from ml.reporting import generate_pdf_report
from ml.taxonomy import DOMAINS, FAULT_CLASSES
from ml.train import train_custom_domain

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
OUTPUTS_DIR = ARTIFACTS_DIR / "outputs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"
CUSTOM_DATA_DIR = ROOT_DIR / "data" / "custom_domain"

for d in (UPLOADS_DIR, OUTPUTS_DIR, REPORTS_DIR, SAMPLES_DIR, CUSTOM_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FaultSense AI — Multi-Domain Deep Learning Fault Inspection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

db = InspectionDB(ARTIFACTS_DIR / "faultsense.db")
engine = FaultSenseEngine(ROOT_DIR)

# Active model selection & custom training state
STATE_LOCK = threading.Lock()
APP_STATE: Dict[str, Any] = {
    "active_model": "builtin",
    "training_status": {
        "state": "idle",  # idle | running | completed | error
        "epoch": 0,
        "total_epochs": 0,
        "best_val_acc": 0.0,
        "history": [],
        "class_names": [],
        "error": None,
    },
}


@app.get("/api/health")
def health() -> Dict[str, Any]:
    with STATE_LOCK:
        active_model = APP_STATE["active_model"]
    return {
        "status": "ok",
        "device": str(engine.device),
        "active_model": active_model,
        "has_custom_model": "custom" in engine.models,
        "custom_classes": engine.custom_class_names,
        "builtin_classes": len(FAULT_CLASSES),
    }


@app.get("/api/taxonomy")
def get_taxonomy() -> Dict[str, Any]:
    return {
        "domains": list(DOMAINS.values()),
        "classes": FAULT_CLASSES,
        "custom_classes": engine.custom_class_names,
    }


@app.get("/api/model/metrics")
def get_model_metrics() -> Dict[str, Any]:
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    if not metrics_file.exists():
        return {"available": False, "message": "Metrics not generated yet."}
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    data["available"] = True
    return data


@app.get("/api/samples")
def list_samples() -> Dict[str, Any]:
    # Ensure samples exist even if called standalone
    if not any(SAMPLES_DIR.iterdir()):
        materialize_dataset_and_samples(ROOT_DIR, samples_per_class=20)

    image_samples = []
    for cls in FAULT_CLASSES:
        p = SAMPLES_DIR / f"sample_{cls['id']}.png"
        if p.exists():
            image_samples.append(
                {
                    "id": cls["id"],
                    "filename": p.name,
                    "media_type": "image",
                    "domain": cls["domain"],
                    "label_en": cls["label_en"],
                    "label_ar": cls["label_ar"],
                    "severity_level": cls["severity_level"],
                    "url": f"/artifacts/samples/{p.name}",
                }
            )

    video_meta = [
        ("video_industrial_scratch.mp4", "industrial", "Conveyor Metal Scratch Clip", "فيديو سير إنتاج: خدش معدن", "warning"),
        ("video_road_pothole.mp4", "infrastructure", "Road Dashcam Pothole Clip", "فيديو طريق: حفرة أسفلت", "critical"),
        ("video_safety_hazard_breach.mp4", "safety", "CCTV Hazard Zone Intrusion", "فيديو مراقبة: اختراق منطقة خطر", "critical"),
        ("video_industrial_clean.mp4", "industrial", "Clean Conveyor Surface Clip", "فيديو سير إنتاج: سطح سليم", "pass"),
    ]
    video_samples = []
    for fname, dom, lbl_en, lbl_ar, sev in video_meta:
        vp = SAMPLES_DIR / fname
        if vp.exists():
            video_samples.append(
                {
                    "id": fname,
                    "filename": fname,
                    "media_type": "video",
                    "domain": dom,
                    "label_en": lbl_en,
                    "label_ar": lbl_ar,
                    "severity_level": sev,
                    "url": f"/artifacts/samples/{fname}",
                }
            )

    return {"images": image_samples, "videos": video_samples}


@app.post("/api/inspect/image")
async def inspect_image_endpoint(
    file: Optional[UploadFile] = File(default=None),
    sample_id: Optional[str] = Form(default=None),
    domain_filter: str = Form(default="all"),
    active_model: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    with STATE_LOCK:
        model_mode = active_model or APP_STATE["active_model"]

    insp_id = uuid.uuid4().hex[:12]
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower() or ".png"
        dest = UPLOADS_DIR / f"{insp_id}{ext}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        orig_name = file.filename
    elif sample_id:
        candidate = SAMPLES_DIR / f"sample_{sample_id}.png"
        if not candidate.exists():
            candidate = SAMPLES_DIR / sample_id
        if not candidate.exists():
            raise HTTPException(status_code=404, detail=f"Sample image '{sample_id}' not found.")
        dest = candidate
        orig_name = candidate.name
    else:
        raise HTTPException(status_code=400, detail="Provide either an image file or a sample_id.")

    result = engine.inspect_image(
        image_path=dest,
        out_dir=OUTPUTS_DIR,
        inspection_id=insp_id,
        domain_filter=domain_filter,
        active_model=model_mode,
    )
    saved = db.save_inspection(orig_name, result)
    return saved


@app.post("/api/inspect/video")
async def inspect_video_endpoint(
    file: Optional[UploadFile] = File(default=None),
    sample_id: Optional[str] = Form(default=None),
    domain_filter: str = Form(default="all"),
    sample_frames: int = Form(default=16),
    active_model: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    with STATE_LOCK:
        model_mode = active_model or APP_STATE["active_model"]

    insp_id = uuid.uuid4().hex[:12]
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower() or ".mp4"
        dest = UPLOADS_DIR / f"{insp_id}{ext}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        orig_name = file.filename
    elif sample_id:
        candidate = SAMPLES_DIR / sample_id
        if not candidate.exists():
            raise HTTPException(status_code=404, detail=f"Sample video '{sample_id}' not found.")
        dest = candidate
        orig_name = candidate.name
    else:
        raise HTTPException(status_code=400, detail="Provide either a video file or a sample_id.")

    result = engine.inspect_video(
        video_path=dest,
        out_dir=OUTPUTS_DIR,
        inspection_id=insp_id,
        sample_frames=max(6, min(24, int(sample_frames))),
        domain_filter=domain_filter,
        active_model=model_mode,
    )
    saved = db.save_inspection(orig_name, result)
    return saved


@app.get("/api/history")
def get_history(
    domain: str = "all",
    severity: str = "all",
    limit: int = 100,
) -> Dict[str, Any]:
    items = db.list_inspections(domain=domain, severity=severity, limit=limit)
    return {"items": items, "count": len(items)}


@app.get("/api/history/{inspection_id}")
def get_history_item(inspection_id: str) -> Dict[str, Any]:
    item = db.get_inspection(inspection_id)
    if not item:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return item


@app.delete("/api/history/{inspection_id}")
def delete_history_item(inspection_id: str) -> Dict[str, Any]:
    ok = db.delete_inspection(inspection_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return {"deleted": True, "id": inspection_id}


@app.get("/api/reports/{inspection_id}/pdf")
def download_pdf_report(inspection_id: str) -> FileResponse:
    record = db.get_inspection(inspection_id)
    if not record:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    pdf_path = REPORTS_DIR / f"FaultSense_Report_{inspection_id}.pdf"
    generate_pdf_report(record, pdf_path)
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"FaultSense_Report_{inspection_id}.pdf",
    )


@app.get("/api/reports/{inspection_id}/json")
def download_json_report(inspection_id: str) -> JSONResponse:
    record = db.get_inspection(inspection_id)
    if not record:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return JSONResponse(
        content=record,
        headers={
            "Content-Disposition": f'attachment; filename="FaultSense_Report_{inspection_id}.json"'
        },
    )


@app.post("/api/model/switch")
def switch_model(mode: str = Form(...)) -> Dict[str, Any]:
    if mode not in ("builtin", "custom"):
        raise HTTPException(status_code=400, detail="mode must be 'builtin' or 'custom'")
    if mode == "custom" and "custom" not in engine.models:
        raise HTTPException(status_code=400, detail="No custom trained model available yet.")
    with STATE_LOCK:
        APP_STATE["active_model"] = mode
    return {"active_model": mode, "custom_classes": engine.custom_class_names}


@app.get("/api/train/status")
def get_training_status() -> Dict[str, Any]:
    with STATE_LOCK:
        return dict(APP_STATE["training_status"])


def _run_custom_job(samples, class_names, epochs: int, lr: float, backbone: str) -> None:
    try:
        def _cb(progress: Dict[str, Any]) -> None:
            with STATE_LOCK:
                APP_STATE["training_status"].update(
                    {
                        "state": "running",
                        "epoch": progress["epoch"],
                        "total_epochs": progress["total_epochs"],
                        "best_val_acc": progress["best_val_acc"],
                        "history": progress["history"],
                        "class_names": class_names,
                    }
                )

        res = train_custom_domain(
            custom_dir=CUSTOM_DATA_DIR,
            ckpt_out_path=engine.custom_ckpt,
            class_names=class_names,
            samples=samples,
            epochs=epochs,
            lr=lr,
            backbone=backbone,
            progress_cb=_cb,
        )
        engine.load_custom_if_available()
        with STATE_LOCK:
            APP_STATE["active_model"] = "custom"
            APP_STATE["training_status"].update(
                {
                    "state": "completed",
                    "epoch": epochs,
                    "total_epochs": epochs,
                    "best_val_acc": res["best_val_acc"],
                    "history": res["history"],
                    "class_names": class_names,
                    "error": None,
                }
            )
    except Exception as exc:
        with STATE_LOCK:
            APP_STATE["training_status"].update({"state": "error", "error": str(exc)})


@app.post("/api/train/custom")
async def start_custom_training(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(default=[]),
    labels_json: str = Form(default="[]"),
    epochs: int = Form(default=6),
    lr: float = Form(default=0.0004),
    backbone: str = Form(default="mobilenet_v3_small"),
    use_demo_custom_pack: bool = Form(default=False),
) -> Dict[str, Any]:
    with STATE_LOCK:
        if APP_STATE["training_status"]["state"] == "running":
            raise HTTPException(status_code=409, detail="A custom training job is already running.")

    # Reset custom workspace
    if CUSTOM_DATA_DIR.exists():
        shutil.rmtree(CUSTOM_DATA_DIR)
    CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if use_demo_custom_pack or not files:
        # Build a clean 3-class custom demo dataset from existing multi-domain textures
        from PIL import Image
        from ml.dataset import synthesize_image

        demo_map = {
            "clean_part": "industrial_normal",
            "surface_scratch": "industrial_scratch",
            "surface_crack": "industrial_crack",
        }
        for cname, synth_id in demo_map.items():
            cdir = CUSTOM_DATA_DIR / cname
            cdir.mkdir(parents=True, exist_ok=True)
            for i in range(18):
                rgb = synthesize_image(synth_id, size=224, seed=42_000 + i + hash(cname) % 500)
                Image.fromarray(rgb).save(cdir / f"{cname}_{i:02d}.png")
    else:
        try:
            labels = json.loads(labels_json)
        except Exception:
            labels = []
        if len(labels) != len(files):
            raise HTTPException(
                status_code=400,
                detail="labels_json length must match uploaded files length.",
            )
        for idx, (uf, lbl) in enumerate(zip(files, labels)):
            safe_lbl = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(lbl).strip()) or "class_a"
            cdir = CUSTOM_DATA_DIR / safe_lbl
            cdir.mkdir(parents=True, exist_ok=True)
            ext = Path(uf.filename or "img.png").suffix.lower() or ".png"
            dest = cdir / f"sample_{idx:03d}{ext}"
            with dest.open("wb") as f:
                shutil.copyfileobj(uf.file, f)

    samples, class_names = load_custom_image_folder(CUSTOM_DATA_DIR)
    if len(class_names) < 2 or len(samples) < 4:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 classes and 4 total images to train a custom classifier.",
        )

    with STATE_LOCK:
        APP_STATE["training_status"] = {
            "state": "running",
            "epoch": 0,
            "total_epochs": epochs,
            "best_val_acc": 0.0,
            "history": [],
            "class_names": class_names,
            "error": None,
        }

    background_tasks.add_task(_run_custom_job, samples, class_names, epochs, lr, backbone)
    return {"started": True, "class_names": class_names, "sample_count": len(samples)}


# Serve built React frontend if available
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
