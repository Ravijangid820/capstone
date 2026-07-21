"""Interactive Web UI Server for Personalized Federated Learning Brain Tumor Segmentation.

    uv run python scripts/demo_server.py
    
Launch this server and navigate to http://localhost:8000 to visualize and compare the 2D/3D segmentations of Centralized, Local, FedAvg, and FedBN models in real-time.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path

# Add src/ to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image

from fedbrats.config import Config
from fedbrats.data import load_case, load_index, preprocess
from fedbrats.model import build_model
from fedbrats.train import predict_volume
from fedbrats.metrics import dice_regions

STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "fedbrats" / "static"

# ─── Model Cache ──────────────────────────────────────────────────────────────
# Cache loaded models by (dim, method, hospital) to avoid reloading on every
# request.  For FedBN, the hospital matters because BN weights differ; for all
# other methods, hospital is ignored in the cache key.
_model_cache: dict[tuple, torch.nn.Module] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model(dim: str, method: str, hospital: str):
    """Return a cached model, loading from checkpoint on first access."""
    cache_key = (dim, method, hospital if method == "fedbn" else "_global")
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    cfg = Config(dim=dim)
    model = build_model(cfg)
    run_id = cfg.run_id(method)
    path = cfg.paths.runs / run_id / "checkpoints" / "final.pt"

    if not path.exists():
        return None  # caller should handle missing checkpoint

    ckpt = torch.load(path, map_location="cpu")
    if method == "fedbn":
        global_w = ckpt["global"]
        bn_state = ckpt["bn"].get(hospital, {})
        model.load_state_dict({**global_w, **bn_state})
    elif isinstance(ckpt, dict) and "global" in ckpt:
        model.load_state_dict(ckpt["global"])
    else:
        model.load_state_dict(ckpt)

    model.to(_device)
    model.eval()
    _model_cache[cache_key] = model
    print(f"  [cache] loaded model: {cache_key}")
    return model


class DemoHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path_str = url.path

        # Routing static assets
        if path_str == "/" or path_str == "/index.html":
            self.serve_file(STATIC_DIR / "index.html", "text/html")
        elif path_str == "/style.css":
            self.serve_file(STATIC_DIR / "style.css", "text/css")
        elif path_str == "/app.js":
            self.serve_file(STATIC_DIR / "app.js", "application/javascript")
        elif path_str == "/marching_cubes.js":
            self.serve_file(STATIC_DIR / "marching_cubes.js", "application/javascript")
        elif path_str.startswith("/vendor/") and path_str.endswith(".js"):
            # Serve bundled vendor libraries (Three.js, OrbitControls)
            vendor_file = path_str.lstrip("/")
            self.serve_file(STATIC_DIR / vendor_file, "application/javascript")
        elif path_str == "/wasm_marching_cubes.wasm":
            self.serve_file(STATIC_DIR / "wasm_marching_cubes.wasm", "application/wasm")
        elif path_str == "/api/cases":
            self.handle_api_cases()
        elif path_str == "/api/view":
            self.handle_api_view(url.query)
        elif path_str == "/api/health":
            self.send_json({"status": "ok", "device": str(_device), "cached_models": len(_model_cache)})
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/predict":
            self.handle_api_predict()
        elif url.path == "/api/mesh":
            self.handle_api_mesh()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def img_to_base64(self, arr: np.ndarray, mode="RGB") -> str:
        img = Image.fromarray(arr, mode=mode)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def create_overlay(self, mri_slice: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mri_rgb = np.stack([mri_slice] * 3, axis=-1)
        wt_color = np.array([16, 185, 129], dtype=np.uint8)
        tc_color = np.array([59, 130, 246], dtype=np.uint8)
        et_color = np.array([236, 72, 153], dtype=np.uint8)

        alpha = 0.45
        overlay = mri_rgb.copy()

        # Apply WT (channel 0)
        overlay = np.where(mask[0][..., None] == 1, (overlay * (1 - alpha) + wt_color * alpha).astype(np.uint8), overlay)
        # Apply TC (channel 1)
        overlay = np.where(mask[1][..., None] == 1, (overlay * (1 - alpha) + tc_color * alpha).astype(np.uint8), overlay)
        # Apply ET (channel 2)
        overlay = np.where(mask[2][..., None] == 1, (overlay * (1 - alpha) + et_color * alpha).astype(np.uint8), overlay)

        return overlay

    def handle_api_cases(self):
        try:
            cfg = Config()
            index = load_index(cfg)
            self.send_json(index)
        except Exception as e:
            self.send_error(500, str(e))

    def handle_api_view(self, query_str):
        try:
            params = urllib.parse.parse_qs(query_str)
            case_id = params.get("case_id", [""])[0]
            slice_idx = int(params.get("slice_idx", [80])[0])
            modality = params.get("modality", ["flair"])[0]
            hospital = params.get("hospital", ["None"])[0]

            if not case_id:
                self.send_error(400, "Missing case_id")
                return

            cfg = Config()

            # Load raw volume
            mods, seg = load_case(cfg.paths.data_root, case_id)

            # Apply scanner shift on-the-fly if selected
            if hospital != "None":
                from fedbrats.shift import apply_shift
                mods_shifted = apply_shift(mods, hospital, cfg.seed)
            else:
                mods_shifted = mods

            # Slice modality
            mod_idx = ["flair", "t1", "t1ce", "t2"].index(modality)
            mri_slice = mods_shifted[mod_idx, :, :, slice_idx]

            # Normalize to 0-255
            v_min, v_max = mri_slice.min(), mri_slice.max()
            if v_max > v_min:
                mri_norm = (255 * (mri_slice - v_min) / (v_max - v_min)).astype(np.uint8)
            else:
                mri_norm = np.zeros_like(mri_slice, dtype=np.uint8)

            # Ground truth overlay
            from fedbrats.data import labels_to_regions
            regions = labels_to_regions(seg)
            gt_slice = regions[:, :, :, slice_idx]

            # Base64 images
            mri_b64 = self.img_to_base64(mri_norm, mode="L")
            gt_overlay = self.create_overlay(mri_norm, gt_slice)
            gt_b64 = self.img_to_base64(gt_overlay, mode="RGB")

            self.send_json({
                "mri_base64": mri_b64,
                "gt_base64": gt_b64
            })
        except Exception as e:
            self.send_error(500, str(e))

    def handle_api_predict(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))

            case_id = req.get("case_id")
            dim = req.get("dim", "2d")
            method = req.get("method", "fedbn")
            hospital = req.get("hospital", "H4")
            slice_idx = req.get("slice_idx", 80)

            if not case_id:
                self.send_error(400, "Missing case_id")
                return

            cfg = Config(dim=dim)

            # Load model from cache (or checkpoint on first access)
            model = _get_model(dim, method, hospital)
            if model is None:
                self.send_json({
                    "dice": {"wt": 0.0, "tc": 0.0, "et": 0.0},
                    "pred_base64": "",
                    "error": f"Model checkpoint not found for {method} {dim}. Please train the model first."
                })
                return

            # Load raw volume & preprocess on-the-fly
            mods, seg = load_case(cfg.paths.data_root, case_id)
            h_shift = hospital if hospital != "None" else None
            x, y, _ = preprocess(mods, seg, hospital=h_shift, seed=cfg.seed, clip=cfg.clip_sigma)

            # Predict volume
            pred = predict_volume(model, x, cfg, _device)

            # Extract slice
            pred_slice = pred[:, :, :, slice_idx]
            gt_slice = y[:, :, :, slice_idx]

            # Compute Dice
            dice = dice_regions(pred_slice, gt_slice)

            # Normalize preprocessed slice modality
            mod_idx = ["flair", "t1", "t1ce", "t2"].index(req.get("modality", "flair"))
            mri_slice = x[mod_idx, :, :, slice_idx]

            v_min, v_max = mri_slice.min(), mri_slice.max()
            if v_max > v_min:
                mri_norm = (255 * (mri_slice - v_min) / (v_max - v_min)).astype(np.uint8)
            else:
                mri_norm = np.zeros_like(mri_slice, dtype=np.uint8)

            # Generate overlay
            pred_overlay = self.create_overlay(mri_norm, pred_slice)
            pred_b64 = self.img_to_base64(pred_overlay, mode="RGB")

            self.send_json({
                "dice": dice,
                "pred_base64": pred_b64
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    def handle_api_mesh(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))

            case_id = req.get("case_id")
            dim = req.get("dim", "2d")
            method = req.get("method", "fedbn")
            hospital = req.get("hospital", "H4")

            if not case_id:
                self.send_error(400, "Missing case_id")
                return

            cfg = Config(dim=dim)

            # Load model from cache (or checkpoint on first access)
            model = _get_model(dim, method, hospital)
            if model is None:
                self.send_json({
                    "error": f"Model checkpoint not found for {method} {dim}."
                })
                return

            # Load raw volume & preprocess on-the-fly
            mods, seg = load_case(cfg.paths.data_root, case_id)
            h_shift = hospital if hospital != "None" else None
            x, y, brain = preprocess(mods, seg, hospital=h_shift, seed=cfg.seed, clip=cfg.clip_sigma)

            # Predict volume
            pred = predict_volume(model, x, cfg, _device)

            # Package volumes as base64 byte arrays for client-side JS marching cubes.
            # Binary masks (0/1) are scaled to (0/255) so the JS isoLevel=128 threshold works.

            self.send_json({
                "shape": list(brain.shape),
                "brain": base64.b64encode((brain.astype(np.uint8) * 255).tobytes()).decode("utf-8"),
                "wt": base64.b64encode((pred[0].astype(np.uint8) * 255).tobytes()).decode("utf-8"),
                "tc": base64.b64encode((pred[1].astype(np.uint8) * 255).tobytes()).decode("utf-8"),
                "et": base64.b64encode((pred[2].astype(np.uint8) * 255).tobytes()).decode("utf-8")
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))



class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads so the UI doesn't freeze during inference."""
    daemon_threads = True


def run_server(port=8000):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, DemoHTTPRequestHandler)
    print(f"Starting brain tumor segmentation demo server on http://localhost:{port}")
    print(f"  Device: {_device}")
    print(f"  Static: {STATIC_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
