"""Cancellable worker jobs; torch and geoai are never imported beside pycolmap."""
from __future__ import annotations

import json
import queue
import subprocess
import tempfile
import threading
from pathlib import Path

from . import environment
from .access import redact


def run(request: dict, progress=None, python=None) -> dict:
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    python = python or environment.worker_python("geoai")
    messages = queue.Queue()
    with tempfile.TemporaryFile(mode="w+b") as log:
        proc = subprocess.Popen([str(python), "-u", str(Path(__file__).with_name("worker.py"))],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                env=environment.clean_environment(), close_fds=False)

        def read():
            try:
                for line in proc.stdout:
                    try:
                        messages.put(json.loads(line))
                    except ValueError:
                        messages.put({"ok": False, "error": "Invalid GeoAI worker response"})
            finally:
                messages.put(None)

        threading.Thread(target=read, daemon=True).start()
        try:
            proc.stdin.write((json.dumps(request) + "\n").encode())
            proc.stdin.flush()
            result = None
            while True:
                progress.check()
                try:
                    item = messages.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if "progress" in item:
                    progress(float(item["progress"]), item["message"])
                elif item.get("ok"):
                    result = item["result"]
                else:
                    raise RuntimeError(redact(item.get("error", "GeoAI worker failed")))
            code = proc.wait(timeout=10)
            if code or result is None:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 3000))
                raise RuntimeError(f"GeoAI worker exited ({code}): " + redact(log.read().decode("utf-8", "replace")))
            return result
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for stream in (proc.stdin, proc.stdout):
                stream.close()
