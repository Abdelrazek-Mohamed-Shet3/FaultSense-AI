"""End-to-end verification tests for FaultSense-DL backend + PyTorch inference."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app


def run_tests() -> None:
    client = TestClient(app)

    # 1. Health & Taxonomy & Metrics
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    health = r.json()
    assert health["status"] == "ok"
    assert health["builtin_classes"] == 10

    r = client.get("/api/model/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["available"] is True
    assert m["test_accuracy"] >= 0.90

    r = client.get("/api/samples")
    assert r.status_code == 200
    samples = r.json()
    assert len(samples["images"]) == 10
    assert len(samples["videos"]) == 4

    # 2. Test Image Inspection across all 3 core domains
    for sample_id, expected_domain in [
        ("industrial_crack", "industrial"),
        ("industrial_pcb_defect", "industrial"),
        ("infra_pothole", "infrastructure"),
        ("infra_corrosion", "infrastructure"),
        ("safety_no_helmet", "safety"),
        ("safety_hazard_zone", "safety"),
    ]:
        resp = client.post(
            "/api/inspect/image",
            data={"sample_id": sample_id, "domain_filter": "all"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["predicted_class"] == sample_id, f"Expected {sample_id}, got {body['predicted_class']}"
        assert body["domain"] == expected_domain
        assert body["confidence"] > 0.70
        assert (ROOT / body["assets"]["gradcam"].lstrip("/")).exists()
        assert (ROOT / body["assets"]["localized"].lstrip("/")).exists()

        # Verify PDF & JSON report generation
        pdf_r = client.get(f"/api/reports/{body['id']}/pdf")
        assert pdf_r.status_code == 200
        assert pdf_r.headers["content-type"] == "application/pdf"
        assert len(pdf_r.content) > 2000

        json_r = client.get(f"/api/reports/{body['id']}/json")
        assert json_r.status_code == 200

    # 3. Test Video Temporal Inspection
    for vid_id, expected_cls in [
        ("video_industrial_scratch.mp4", "industrial_scratch"),
        ("video_road_pothole.mp4", "infra_pothole"),
        ("video_safety_hazard_breach.mp4", "safety_hazard_zone"),
        ("video_industrial_clean.mp4", "industrial_normal"),
    ]:
        vresp = client.post(
            "/api/inspect/video",
            data={"sample_id": vid_id, "domain_filter": "all", "sample_frames": "12"},
        )
        assert vresp.status_code == 200, vresp.text
        vbody = vresp.json()
        assert vbody["media_type"] == "video"
        assert vbody["predicted_class"] == expected_cls, f"Expected {expected_cls}, got {vbody['predicted_class']}"
        assert len(vbody["timeline"]) == 12
        assert len(vbody["keyframes"]) == 3
        mp4_path = ROOT / vbody["assets"]["annotated_video"].lstrip("/")
        assert mp4_path.exists() and mp4_path.stat().st_size > 1000

    # 4. History & Frontend root
    hr = client.get("/api/history")
    assert hr.status_code == 200
    assert hr.json()["count"] >= 10

    root_r = client.get("/")
    assert root_r.status_code == 200
    assert "FaultSense AI" in root_r.text

    print("ALL E2E TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
