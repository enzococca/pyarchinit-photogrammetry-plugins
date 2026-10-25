"""Standalone supervised GeoAI worker. Only this process imports torch/geoai.

One JSON request on stdin; JSON progress/result lines on stdout. Library output goes to stderr.
PPG_GEOAI_FAKE=1 uses colour centroids for integration tests, never an actual neural model.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from dataset import IGNORE, check_raster, read_manifest
from inputs import RasterStack, channel_count, normalise


def tile_image(data, meta, *, scaled=True):
    """The saved training transform, with nodata neutral in the model's feature space."""
    image = np.asarray(data["image"], dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != channel_count(meta):
        raise ValueError("Training tile channels do not match the dataset")
    if scaled:
        image = normalise(image, meta)
    if "valid" in data:
        image[:, ~data["valid"].astype(bool)] = 0
    return np.ascontiguousarray(image)


def scores(matrix: np.ndarray, classes: list) -> dict:
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - np.diag(matrix)
    iou = np.divide(np.diag(matrix), union, out=np.full(len(classes), np.nan), where=union > 0)
    return {"iou": {c["name"]: float(iou[i]) if np.isfinite(iou[i]) else None for i, c in enumerate(classes)},
            "mean_iou": float(np.nanmean(iou)), "confusion": matrix.tolist()}


def device_for(requested):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(req, report):
    source, out = Path(req["dataset"]), Path(req["output"])
    meta = read_manifest(source)
    epochs = int(req.get("epochs", 20))
    batch_size = int(req.get("batch_size", 2))
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    train_rows = [s for s in meta["samples"] if s["split"] == "train"]
    val_rows = [s for s in meta["samples"] if s["split"] == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("Both spatial training and validation sets are required")
    if os.environ.get("PPG_GEOAI_FAKE") != "1" and meta["tile_size"] < 64:
        raise ValueError("U-Net/ResNet34 training requires tiles of at least 64 pixels; "
                         "prepare a new dataset with tile_size >= 64")
    out.mkdir(parents=True, exist_ok=False)
    nclasses = len(meta["classes"])
    if os.environ.get("PPG_GEOAI_FAKE") == "1":
        sums, counts = np.zeros((nclasses, channel_count(meta))), np.zeros(nclasses)
        for sample in train_rows:
            with np.load(source / sample["file"], allow_pickle=False) as d:
                image = np.moveaxis(tile_image(d, meta, scaled=meta["format"] == 2), 0, -1)
                for cid in range(nclasses):
                    pix = image[d["label"] == cid]
                    sums[cid] += pix.sum(axis=0)
                    counts[cid] += len(pix)
        if (counts == 0).any():
            raise ValueError("Training set lacks a class")
        centers = sums / counts[:, None]
        np.save(out / "fake.npy", centers)
        matrix = np.zeros((nclasses, nclasses), np.int64)
        for sample in val_rows:
            with np.load(source / sample["file"], allow_pickle=False) as d:
                image = np.moveaxis(tile_image(d, meta, scaled=meta["format"] == 2), 0, -1)
                pred = ((image[..., None, :] - centers) ** 2).sum(axis=-1).argmin(axis=-1)
                y = d["label"]
                known = y != IGNORE
                matrix += np.bincount(nclasses * y[known].astype(int) + pred[known],
                                      minlength=nclasses ** 2).reshape(nclasses, nclasses)
        result = scores(matrix, meta["classes"])
        versions, device, backend = {}, "fake", "fake-colour-centroids"
    else:
        import torch
        from geoai.train import get_smp_model
        from importlib.metadata import version
        from torch.utils.data import DataLoader, Dataset

        torch.manual_seed(42)
        device = device_for(req.get("device"))

        class Tiles(Dataset):
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                with np.load(source / self.rows[idx]["file"], allow_pickle=False) as data:
                    return (torch.from_numpy(tile_image(data, meta)),
                            torch.from_numpy(data["label"].astype(np.int64)))

        encoder_weights = "imagenet" if meta["format"] == 1 else None
        model = get_smp_model(architecture="unet", encoder_name="resnet34", encoder_weights=encoder_weights,
                              in_channels=channel_count(meta), classes=nclasses).to(device)
        loader = DataLoader(Tiles(train_rows), batch_size=batch_size, shuffle=True, num_workers=0)
        validation = DataLoader(Tiles(val_rows), batch_size=batch_size, shuffle=False, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
        best = -1.0
        history = []
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for image, label in loader:
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                loss = criterion(model(image), label)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            model.eval()
            matrix = np.zeros((nclasses, nclasses), np.int64)
            with torch.inference_mode():
                for image, label in validation:
                    pred = model(image.to(device)).argmax(dim=1).cpu().numpy()
                    y = label.numpy()
                    known = y != IGNORE
                    matrix += np.bincount(nclasses * y[known] + pred[known],
                                          minlength=nclasses ** 2).reshape(nclasses, nclasses)
            metrics = scores(matrix, meta["classes"])
            history.append({"epoch": epoch + 1, "loss": total_loss / len(loader), **metrics})
            if metrics["mean_iou"] > best:
                best, result = metrics["mean_iou"], {"best_epoch": epoch + 1, **metrics}
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "weights.pt")
            report((epoch + 1) / epochs, f"Epoch {epoch + 1}/{epochs}; validation mean IoU {metrics['mean_iou']:.3f}")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        versions = {"torch": str(torch.__version__), "geoai-py": version("geoai-py"),
                    "segmentation-models-pytorch": version("segmentation-models-pytorch")}
        backend = "geoai-unet"
    model_meta = {k: meta[k] for k in ("format", "classes", "mapping", "field", "bands", "normalisation",
                                      "resolution", "tile_size", "ignore_index", "split", "class_pixels")}
    if meta["format"] == 2:
        model_meta.update({k: meta[k] for k in ("inputs", "channels")})
    model_meta.update({"architecture": "unet", "encoder": "resnet34", "backend": backend,
                       "encoder_weights": "imagenet" if meta["format"] == 1 else None,
                       "device": str(device), "versions": versions, "metrics": result,
                       "dataset": str(source.resolve()), "epochs": epochs, "batch_size": batch_size})
    (out / "model.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")
    return model_meta


def predict(req, report):
    import math

    import rasterio
    from rasterio.windows import Window

    folder, output = Path(req["model"]), Path(req["output"])
    meta = read_manifest(folder, "model.json")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    custom = meta["format"] == 2
    if custom and not req.get("inputs"):
        raise ValueError("Choose the input rasters required by this model; training paths are not reused")
    if not custom and req.get("inputs"):
        raise ValueError("This RGB model needs the original three-band RGB input")
    fake = meta["backend"] == "fake-colour-centroids"
    if fake and os.environ.get("PPG_GEOAI_FAKE") != "1":
        raise ValueError("This is a test model, not a trained GeoAI model")
    if fake:
        centers = np.load(folder / "fake.npy", allow_pickle=False)

        if centers.shape != (len(meta["classes"]), channel_count(meta)):
            raise ValueError("Test model channels do not match its metadata")

        def infer(image, valid):
            if custom:
                image = normalise(image, meta)
                image[:, ~valid] = 0
            image = np.moveaxis(image, 0, -1)
            return ((image[..., None, :] - centers) ** 2).sum(axis=-1).argmin(axis=-1).astype(np.uint8)
    else:
        import torch
        from geoai.train import get_smp_model

        device = device_for(req.get("device"))
        model = get_smp_model(architecture=meta["architecture"], encoder_name=meta["encoder"],
                              encoder_weights=None, in_channels=channel_count(meta), classes=len(meta["classes"]))
        model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        model.to(device).eval()

        def infer(image, valid):
            with torch.inference_mode():
                image = normalise(image, meta)
                image[:, ~valid] = 0
                tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(device)
                return model(tensor).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    tile = int(meta["tile_size"])
    halo = min(32, tile // 4)
    core = tile - 2 * halo
    from contextlib import ExitStack

    with ExitStack() as opened:
        src = opened.enter_context(rasterio.open(req["raster"]))
        stack = (opened.enter_context(RasterStack(req["raster"], req["inputs"], schema=meta["inputs"]))
                 if custom else None)
        resolution = stack.resolution if custom else check_raster(src)
        if not math.isclose(resolution, meta["resolution"], rel_tol=1e-4):
            raise ValueError(f"Model resolution is {meta['resolution']:g} m/px; resample the raster before applying")
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=IGNORE, compress="deflate")
        profile.pop("photometric", None)
        output.parent.mkdir(parents=True, exist_ok=True)
        nrows, ncols = math.ceil(src.height / core), math.ceil(src.width / core)
        with rasterio.open(output, "w", **profile) as dst:
            dst.update_tags(ppg_name="geoai_classes", geoai_model=str(folder.resolve()))
            for row in range(nrows):
                for col in range(ncols):
                    x, y = col * core, row * core
                    w, h = min(core, src.width - x), min(core, src.height - y)
                    win = Window(x - halo, y - halo, tile, tile)
                    if custom:
                        image, valid = stack.read(win)
                    else:
                        image = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
                        valid = (src.read_masks([1, 2, 3], window=win, boundless=True) > 0).all(axis=0)
                    pred = infer(image, valid)[halo:halo + h, halo:halo + w].copy()
                    pred[~valid[halo:halo + h, halo:halo + w]] = IGNORE
                    dst.write(pred, 1, window=Window(x, y, w, h))
                report((row + 1) / nrows, f"GeoAI: {row + 1}/{nrows} tile rows")
    return {"model": str(folder.resolve()), "classes": meta["classes"], "resolution": meta["resolution"],
            "backend": meta["backend"], "output": str(output), "inputs": meta.get("inputs", []),
            "channels": channel_count(meta), "normalisation": meta["normalisation"]}


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)

    def emit(item):
        protocol.write(json.dumps(item) + "\n")

    try:
        req = json.loads(sys.stdin.readline())
        fn = {"train": train, "predict": predict}[req["op"]]
        result = fn(req, lambda fraction, message: emit({"progress": fraction, "message": message}))
        emit({"ok": True, "result": result})
    except Exception as exc:
        traceback.print_exc()
        emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
