"""Tests for the demo API with a toy reference and a fake scorer (no model, no patient data)."""

import io

import numpy as np
import pytest
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.routers.site_check import parse_labels_csv
from fastapi.testclient import TestClient
from PIL import Image

from tbshift.site_check import UNLABELLED, build_reference


class FakeScorer:
    """Logit = mean brightness, so a test controls the score through the image."""

    def score(self, images):
        return np.array([np.asarray(img, dtype=float).mean() / 25.5 - 5.0 for img in images])


@pytest.fixture(scope="module")
def ref():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(2.0, 1.0, 150), rng.normal(-2.0, 1.0, 150)])
    return build_reference(z, np.r_[np.ones(150, int), np.zeros(150, int)], "toy", "toysite")


def _client(tmp_path, reference, scorer):
    settings = Settings(root=tmp_path, frontend_dir=tmp_path / "no_frontend")
    return TestClient(create_app(settings, reference=reference, scorer=scorer))


@pytest.fixture
def client(tmp_path, ref):
    with _client(tmp_path, ref, FakeScorer()) as c:
        yield c


def _png(value: int, size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, value).save(buf, format="PNG")
    return buf.getvalue()


def test_health_reports_what_is_loaded(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["reference"]["tb_labels_needed"] == 22


def test_scores_endpoint_runs_the_check(client):
    rng = np.random.default_rng(1)
    logits = np.r_[rng.normal(2, 1, 20), rng.normal(-2, 1, 20)].tolist()
    body = client.post("/api/site-check/scores", json={"logits": logits}).json()
    assert body["verdict"] == "plausible" and len(body["images"]) == 40
    assert body["images"][0]["label"] is None and body["ground_truth"] is None


def test_scores_endpoint_validates_input(client):
    assert client.post("/api/site-check/scores", json={"logits": []}).status_code == 422
    r = client.post("/api/site-check/scores", json={"logits": [0.1, 0.2], "labels": [1]})
    assert r.status_code == 422
    r = client.post("/api/site-check/scores", json={"logits": [0.1, 0.2], "labels": [1, 3]})
    assert r.status_code == 422


def test_scores_endpoint_accepts_partial_labels(client):
    logits = [3.0] * 25 + [-3.0] * 25
    labels = [1] * 25 + [None] * 25
    body = client.post("/api/site-check/scores", json={"logits": logits, "labels": labels}).json()
    assert body["verdict"] == "recertified" and body["n_labelled_tb"] == 25


def test_image_upload_is_scored_and_labelled(client):
    files = [("images", (f"x{i}.png", _png(80 + i), "image/png")) for i in range(25)]
    csv = "filename,label\nx0.png,1\nx1.png,0\n"
    files.append(("labels_csv", ("labels.csv", csv.encode(), "text/csv")))
    body = client.post("/api/site-check/images", files=files).json()
    assert body["n_images"] == 25 and body["n_labelled"] == 2
    by_name = {im["name"]: im for im in body["images"]}
    assert by_name["x0.png"]["label"] == 1 and by_name["x5.png"]["label"] is None
    assert by_name["x24.png"]["probability"] > by_name["x0.png"]["probability"]


def test_unreadable_image_gives_a_clear_error(client):
    r = client.post("/api/site-check/images", files=[("images", ("bad.png", b"not an image", "image/png"))])
    assert r.status_code == 422 and "bad.png" in r.json()["detail"]


def test_duplicate_names_are_rejected(client):
    files = [("images", ("same.png", _png(10), "image/png"))] * 2
    assert client.post("/api/site-check/images", files=files).status_code == 422


def test_scenarios_list_and_ground_truth(client):
    names = [s["name"] for s in client.get("/api/scenarios").json()]
    assert names == ["similar", "harmless", "harmful", "harmful_labelled"]
    harmful = client.post("/api/site-check/scenario/harmful").json()
    assert harmful["verdict"] == "recalibrate"
    assert harmful["ground_truth"]["source_threshold_sensitivity"] < 0.9
    fixed = client.post("/api/site-check/scenario/harmful_labelled").json()
    assert fixed["verdict"] == "recertified"
    assert fixed["ground_truth"]["in_force_sensitivity"] > harmful["ground_truth"]["in_force_sensitivity"]
    assert client.post("/api/site-check/scenario/nope").status_code == 404


def test_missing_model_and_reference_give_503(tmp_path):
    with _client(tmp_path, None, None) as c:
        assert c.get("/api/health").json()["model_loaded"] is False
        assert c.post("/api/site-check/scores", json={"logits": [0.0] * 30}).status_code == 503
        r = c.post("/api/site-check/images", files=[("images", ("a.png", _png(1), "image/png"))])
        assert r.status_code == 503


def test_results_endpoints(tmp_path, ref):
    ms = tmp_path / "results" / "multi_seed"
    ms.mkdir(parents=True)
    (ms / "p4_per_seed.csv").write_text(
        "source,target,mask,evaluated_on,method,violation_rate,mean_sensitivity,mean_specificity,seed\n"
        "shenzhen,montgomery,none,source_test,certified,0.05,0.93,0.6,0\n"
        "shenzhen,montgomery,none,target,certified,1.0,0.77,0.7,0\n"
        "shenzhen,montgomery,none,target,fixed,0.0,0.8,0.8,0\n", encoding="utf-8")
    (ms / "p2_per_seed.csv").write_text(
        "source,mask,ft_auc_target,seed\nshenzhen,none,0.85,0\n", encoding="utf-8")
    with _client(tmp_path, ref, None) as c:
        assert c.get("/api/results").json()["tables"] == ["p2_per_seed", "p4_per_seed"]
        rows = c.get("/api/results/headline").json()
        assert rows == [{"seed": 0, "target": "montgomery", "control_violation": 0.05,
                         "target_auc": 0.85, "target_violation": 1.0,
                         "target_sensitivity": 0.77, "target_specificity": 0.7}]
        assert c.get("/api/results/secret_file").status_code == 404


def test_parse_labels_csv():
    out = parse_labels_csv("﻿filename,label\nb.png,1\n", ["a.png", "b.png"])
    assert out.tolist() == [UNLABELLED, 1]
    with pytest.raises(ValueError, match="header"):
        parse_labels_csv("a.png,1\n", ["a.png"])
    with pytest.raises(ValueError, match="not one of"):
        parse_labels_csv("filename,label\nz.png,1\n", ["a.png"])
    with pytest.raises(ValueError, match="0 or 1"):
        parse_labels_csv("filename,label\na.png,yes\n", ["a.png"])
