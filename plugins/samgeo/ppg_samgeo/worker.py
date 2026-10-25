"""Objects found by a text on an image, in a process of its own.

PyTorch never runs in the program's process (its OpenMP runtime and pycolmap's abort it when both
start): the plugin runs this file with a Python that has PyTorch - the environment of the AI plugins
or the program's own Python when PyTorch is installed there - and only the requests and the replies
below go between the two. This file imports nothing of the program or of the plugin.

    python -u worker.py

Each message, both ways, is an 8-byte big-endian length followed by a pickle. The requests:

* ``{"op": "init", "engine": "open", "detector": "IDEA-Research/grounding-dino-base",
  "segmenter": "facebook/sam2.1-hiera-small", "device": None, "token": None}`` ->
  ``{"engine", "device", "versions"}``: the models loaded (downloaded from Hugging Face the first
  time) on the device asked, else CUDA, Apple's GPU (MPS, with ``PYTORCH_ENABLE_MPS_FALLBACK`` for
  the few operations Metal lacks) or the processor. While the weights are downloaded, before the
  reply, ``{"download": True, "done", "total"}`` (bytes) comes every second;
* ``{"op": "segment", "image": uint8 (H, W, 3), "prompt": "stone. brick.", "box_threshold": 0.3,
  "text_threshold": 0.25}`` -> ``{"shape": (H, W), "objects": [...]}``, each object
  ``{"box": [x0, y0, x1, y1], "score", "quality", "label", "mask": (row, col, height, width,
  packed bits)}``: the mask cut to its own box (most objects are small in a tile);
* ``{"op": "quit"}``.

Replies are ``{"ok": True, ...}`` or ``{"ok": False, "error": text, "traceback": text}``.

The open engine is the recipe of samgeo's ``LangSAM``: Grounding DINO (``transformers``) finds the
boxes of the phrases of the text, SAM 2.1 (``sam2``) draws the mask in each box. The score of an
object is Grounding DINO's confidence in its box, ``quality`` SAM's own estimate of its mask.

``PPG_SAMGEO_FAKE`` makes a fake engine without PyTorch, for the tests: the bright blobs of the
image (grey above 200) are the objects, the score is their brightness. ``slow`` sleeps on every
image, ``fail`` answers with an error, ``crash`` closes the process, ``xet`` fails to load the models
unless ``HF_HUB_DISABLE_XET`` is set (a download broken off), ``download`` reports a download first.
"""
from __future__ import annotations

import os
import pickle
import struct
import sys
import threading
import time
import traceback

try:
    from .access import redact
except ImportError:
    from access import redact

SAM_BATCH = 16  # boxes given to SAM at once: the masks of a batch are full images until they are cut


def _read(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack(">Q", head)
    return pickle.loads(stream.read(n))


_WRITING = threading.Lock()  # the replies and the reports of the download thread, one message at a time


def _write(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    with _WRITING:
        stream.write(struct.pack(">Q", len(data)))
        stream.write(data)
        stream.flush()


def _files_size(folder: str) -> int:
    """The bytes of the files under ``folder`` (the links of the snapshots left out)."""
    total = 0
    for root, _dirs, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            if not os.path.islink(path):
                try:
                    total += os.path.getsize(path)
                except OSError:
                    pass
    return total


def download(repos, report, token=None, every: float = 1.0) -> None:
    """The files the models need, ``repos`` as ``(repo id, file patterns)``, downloaded from Hugging Face
    into its cache before the models are loaded, ``report(done, total)`` in bytes every ``every`` seconds
    while they come: the bytes huggingface_hub counts as they arrive, or those written into the cache
    (Xet writes a file in blocks of 100 MB). What is in the cache already is not downloaded again;
    without the internet, or with a huggingface_hub that cannot say the sizes beforehand, nothing is
    reported and the models find their files as usual."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    extra = {"token": token} if token else {}
    try:
        planned = [(repo, patterns, snapshot_download(repo, allow_patterns=patterns, dry_run=True, **extra))
                   for repo, patterns in repos]
    except Exception:  # noqa: BLE001 - offline, or an older huggingface_hub without dry_run
        return
    total = sum(int(f.file_size or 0) for _r, _p, files in planned for f in files if f.will_download)
    if total <= 0:
        return  # all in the cache
    folders = {os.path.dirname(os.path.dirname(os.path.dirname(files[0].local_path)))
               for _r, _p, files in planned if files}
    before = sum(_files_size(f) for f in folders)
    arrived = [0]  # the bytes of huggingface_hub's bar "Downloading bytes"

    class Bytes(tqdm):
        def __init__(self, *args, **kwargs):
            counted = kwargs.get("unit") == "B" and str(kwargs.get("desc") or "").startswith("Downloading")
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self.ppg_counted = counted

        def update(self, n=1):
            if self.ppg_counted and n:
                arrived[0] += n
            return super().update(n)

    stop = threading.Event()

    def watch():
        while not stop.wait(every):
            written = sum(_files_size(f) for f in folders) - before
            report(min(total, max(0, arrived[0], written)), total)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    report(0, total)
    try:
        for repo, patterns, _files in planned:
            try:
                snapshot_download(repo, allow_patterns=patterns, tqdm_class=Bytes, **extra)
            except TypeError:  # a huggingface_hub without tqdm_class: the cache alone says how far
                snapshot_download(repo, allow_patterns=patterns, **extra)
    finally:
        stop.set()
        watcher.join(timeout=every + 1)
    report(total, total)


def _cut(mask):
    """A boolean mask cut to the box of its pixels: ``(row, col, height, width, packed bits)``,
    None when it is empty."""
    import numpy as np

    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    crop = np.ascontiguousarray(mask[r0:r1, c0:c1])
    return (r0, c0, r1 - r0, c1 - c0, np.packbits(crop).tobytes())


class Fake:
    """The bright blobs of the image are the objects (tests, without PyTorch)."""

    engine = "fake"
    device = "fake"

    def __init__(self, mode: str):
        self.mode = mode
        self.versions = {"fake": "1"}

    def segment(self, image, prompt, box_threshold, text_threshold):
        from scipy import ndimage

        if self.mode == "slow":
            time.sleep(2.0)
        elif self.mode == "fail":
            raise RuntimeError("the fake engine refused the image")
        elif self.mode == "crash":
            os._exit(3)
        grey = image[..., :3].astype(float).mean(axis=2)
        labels, n = ndimage.label(grey > 200)
        label = prompt.split(".")[0].strip()
        objects = []
        for k, sl in enumerate(ndimage.find_objects(labels), 1):
            if sl is None:
                continue
            mask = labels == k
            if mask.sum() < 4:
                continue
            score = float(grey[mask].mean() / 255.0)
            if score < box_threshold:
                continue
            objects.append({"box": [float(sl[1].start), float(sl[0].start), float(sl[1].stop), float(sl[0].stop)],
                            "score": score, "quality": 1.0, "label": label, "mask": _cut(mask)})
        return objects


class Open:
    """Grounding DINO finds the boxes of the text, SAM 2.1 draws the mask in each box."""

    engine = "open"

    def __init__(self, detector: str, segmenter: str, device: str | None, token: str | None, report=None):
        try:  # the weights downloaded first, saying how far they are
            from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

            checkpoint = HF_MODEL_ID_TO_FILENAMES[segmenter][1]
        except (ImportError, KeyError):
            checkpoint = "*.pt"
        if report is not None:
            download([(detector, ["*.json", "*.txt", "model.safetensors"]), (segmenter, [checkpoint])], report, token)
        import torch
        import transformers
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if device is None:
            mps = getattr(torch.backends, "mps", None)
            device = "cuda" if torch.cuda.is_available() else "mps" if mps is not None and mps.is_available() else "cpu"
        self.device = device
        extra = {"token": token} if token else {}
        self.processor = AutoProcessor.from_pretrained(detector, **extra)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector, **extra).to(device).eval()
        self.predictor = SAM2ImagePredictor.from_pretrained(segmenter, device=device)
        from importlib.metadata import version

        # plain strings: torch.__version__ is a class of torch, which the program cannot unpickle
        self.versions = {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
                         "sam2": str(version("sam2"))}

    def _boxes(self, image, prompt, box_threshold, text_threshold):
        import inspect

        import torch
        from PIL import Image

        h, w = image.shape[:2]
        inputs = self.processor(images=Image.fromarray(image), text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        post = self.processor.post_process_grounded_object_detection
        # transformers 4.51 renamed box_threshold to threshold
        key = "threshold" if "threshold" in inspect.signature(post).parameters else "box_threshold"
        result = post(outputs, inputs["input_ids"], text_threshold=text_threshold, target_sizes=[(h, w)],
                      **{key: box_threshold})[0]
        labels = result.get("text_labels", result.get("labels", []))
        return (result["boxes"].float().cpu().numpy(), result["scores"].float().cpu().numpy(),
                [str(x) for x in labels])

    def segment(self, image, prompt, box_threshold, text_threshold):
        import numpy as np
        import torch

        boxes, scores, labels = self._boxes(image, prompt, box_threshold, text_threshold)
        if not len(boxes):
            return []
        h, w = image.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        self.predictor.set_image(image)
        objects = []
        for start in range(0, len(boxes), SAM_BATCH):
            batch = boxes[start:start + SAM_BATCH]
            with torch.inference_mode():
                masks, quality, _ = self.predictor.predict(box=batch, multimask_output=False)
            masks = np.asarray(masks).reshape(len(batch), -1, h, w)[:, 0] > 0
            quality = np.asarray(quality).reshape(len(batch), -1)[:, 0]
            for k, mask in enumerate(masks):
                cut = _cut(mask)
                if cut is None:
                    continue
                i = start + k
                objects.append({"box": [float(v) for v in boxes[i]], "score": float(scores[i]),
                                "quality": float(quality[k]), "label": labels[i] if i < len(labels) else "",
                                "mask": cut})
        return objects


def main() -> int:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # what PyTorch and Hugging Face print goes to stderr, not into the protocol
    engine = None
    while True:
        req = _read(proto_in)
        if req is None or req.get("op") == "quit":
            return 0
        try:
            op = req["op"]
            if op == "init":
                fake = os.environ.get("PPG_SAMGEO_FAKE")
                if fake == "xet" and not os.environ.get("HF_HUB_DISABLE_XET"):  # a download broken off
                    raise RuntimeError("Task error: File reconstruction error: CAS Client Error: Format error: I/O "
                                       "error: error decoding response body")
                def report(done, total):
                    _write(proto_out, {"download": True, "done": int(done), "total": int(total)})

                if fake == "download":  # what a first run says while the weights come
                    for k in range(4):
                        report(k * 250_000_000, 1_000_000_000)
                if fake:
                    engine = Fake(fake)
                elif req.get("engine", "open") == "open":
                    engine = Open(req.get("detector") or "IDEA-Research/grounding-dino-base",
                                  req.get("segmenter") or "facebook/sam2.1-hiera-small", req.get("device"),
                                  req.get("token"), report)
                elif req.get("engine") == "sam3":
                    try:
                        from .worker_sam3 import Sam3
                    except ImportError:
                        from worker_sam3 import Sam3
                    engine = Sam3(req.get("device"), req.get("token"), report)
                else:
                    raise ValueError(f"unknown engine {req.get('engine')!r}")
                reply = {"engine": engine.engine, "device": engine.device, "versions": engine.versions}
            elif engine is None:
                raise RuntimeError("init first")
            elif op == "segment":
                image = req["image"]
                objects = engine.segment(image, req["prompt"], float(req.get("box_threshold", 0.3)),
                                         float(req.get("text_threshold", 0.25)))
                reply = {"shape": tuple(image.shape[:2]), "objects": objects}
            else:
                raise ValueError(f"unknown op {op!r}")
            _write(proto_out, {"ok": True, **reply})
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the plugin
            _write(proto_out, {"ok": False, "error": redact(f"{type(exc).__name__}: {exc}", req.get("token")),
                                     "traceback": redact(traceback.format_exc(), req.get("token"))})


if __name__ == "__main__":
    sys.exit(main())
