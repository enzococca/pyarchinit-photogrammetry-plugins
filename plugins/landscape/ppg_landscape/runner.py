"""The worker in a process of its own: the job written beside the input arrays, the messages read
while it runs (the progress bar moves, cancelling kills it at once, even inside a long step), the
result or the reason of a failure.

The process starts as every process of the program does: the absolute path of its Python, the
file descriptors inherited and no working directory (posix_spawn on macOS: a fork after SpatiaLite
was loaded can die in the child), without the variables of the program's Python.

A run that does not finish leaves nothing: the worker writes in ``<output>.partial`` and renames
the folder at the end; after a failure or a cancel the runner removes both the partial folder and
an output folder the worker may have renamed just before it was stopped.
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path

from . import environment
from .texts import phase, t, years

WORKER = Path(__file__).with_name("worker.py")
PREFIX = "@ppg-landscape "
POLL = 0.2  # seconds between two looks at the cancel button while the worker says nothing
KEEP_LINES = 80  # lines of Landlab's own output kept for the report of a failure


class WorkerError(RuntimeError):
    """The worker failed: ``phase`` says where, ``output`` holds Landlab's last lines, ``details``
    the traceback of the worker."""

    def __init__(self, message: str, phase: str | None = None, output=(), details: str = ""):
        super().__init__(message)
        self.phase, self.output, self.details = phase, list(output), details


def parse(line: str) -> dict | None:
    """The message of a line of the worker, None for Landlab's own lines."""
    i = line.find(PREFIX)
    if i < 0:
        return None
    try:
        msg = json.loads(line[i + len(PREFIX):])
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def phase_text(msg: dict) -> str:
    """What the worker is doing, in the language of the interface."""
    name = str(msg.get("phase") or "")
    if name == "run" and msg.get("year") is not None and msg.get("years"):
        return "Landlab: " + phase("run").format(years(msg["year"]), years(msg["years"]))
    return "Landlab: " + phase(name)


def hint_for(text: str, output=()) -> str:
    """A suggestion for a failure the user can do something about."""
    low = (text + "\n" + "\n".join(output)).lower()
    if "memoryerror" in low or "unable to allocate" in low or "cannot allocate memory" in low:
        return t("Memoria insufficiente: scegli una cella più grande (meno celle nella griglia).",
                 "Not enough memory: choose a larger cell (fewer cells in the grid).")
    if "no module named" in low and "landlab" in low:
        return t("Landlab non c'è più nell'ambiente: installa di nuovo Landlab.",
                 "Landlab is no longer in the environment: install Landlab again.")
    if "became nan" in low or "floatingpointerror" in low:
        return t("La simulazione è diventata instabile: riduci il passo (dt) o i coefficienti di erosione.",
                 "The simulation became unstable: reduce the step (dt) or the erosion coefficients.")
    if "fewer than 9 cells" in low or "no cell inside" in low:
        return t("Il DEM ha troppo poche celle con un valore per questa cella di simulazione: "
                 "scegline una più piccola.",
                 "The DEM has too few cells with a value for this simulation cell: choose a smaller one.")
    return ""


def _crash_text(code: int, last: dict | None) -> str:
    where = t(" durante «{0}»", " during '{0}'").format(phase_text(last)[len("Landlab: "):]) if last else ""
    return t("Landlab si è chiuso inaspettatamente{0} (codice {1}): nessun risultato è stato scritto.",
             "Landlab closed unexpectedly{0} (code {1}): no result was written.").format(where, code)


def _clean(output: Path | None, existed: bool) -> None:
    """What a run that did not finish may have left: the partial folder, and the output folder when
    it was not there before (renamed by the worker just before it was stopped)."""
    if output is None:
        return
    shutil.rmtree(output.with_name(output.name + ".partial"), ignore_errors=True)
    if not existed:
        shutil.rmtree(output, ignore_errors=True)


def run(job: dict, folder: Path, progress=None, python: Path | None = None, script: Path | None = None) -> dict:
    """Run ``job`` (see :mod:`.worker`) with ``python`` (the one of :func:`environment.worker_python`
    by default) and return its result. ``folder`` holds the job file. Raises :class:`WorkerError`,
    :class:`environment.LandlabMissing` without a Python for it, ``Cancelled`` when the user stops it;
    in every one of those cases no output is left."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    python = Path(python).absolute() if python is not None else environment.worker_python()
    script = Path(script).absolute() if script is not None else WORKER.absolute()
    folder = Path(folder).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    output = Path(job["output"]).absolute() if job.get("output") else None
    existed = output is not None and output.exists()
    job_path = folder / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    progress(0.0, phase_text({"phase": "start"}))
    # -I: neither the plugin's folder nor PYTHON* variables on the worker's path (it loads the engine by its path)
    proc = subprocess.Popen([str(python), "-I", "-u", str(script), str(job_path)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=environment.clean_environment(), close_fds=False)
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for raw in proc.stdout:
                lines.put(raw.rstrip("\r\n"))
        except (OSError, ValueError):  # the pipe closed under it
            pass
        finally:
            lines.put(None)

    reader = threading.Thread(target=pump, name="landscape worker output", daemon=True)
    reader.start()
    output_lines: deque = deque(maxlen=KEEP_LINES)
    result = error = hello = last = None
    finished = False
    try:
        try:
            while True:
                try:
                    line = lines.get(timeout=POLL)
                except queue.Empty:
                    progress.check()  # the user may cancel while a long step runs and the worker says nothing
                    continue
                if line is None:
                    break
                progress.check()
                msg = parse(line)
                if msg is None:
                    if line.strip():
                        output_lines.append(line)
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    last = msg
                    progress(min(max(float(msg.get("fraction", 0.0)), 0.0), 1.0), phase_text(msg))
                elif kind == "hello":
                    hello = msg
                elif kind == "result":
                    result = msg.get("result")
                elif kind == "error":
                    error = msg
            code = proc.wait()
        except BaseException:  # cancelled (or the program stopping): the worker goes at once
            proc.kill()
            proc.wait()
            raise
        finally:
            reader.join(timeout=5)
            try:
                proc.stdout.close()
            except OSError:
                pass
        if error is not None:
            text = str(error.get("error") or "").strip()
            message = f"Landlab ({phase(str(error.get('phase') or 'start'))}): {text}"
            hint = hint_for(text, output_lines)
            details = str(error.get("traceback") or "") + ("\n" + "\n".join(output_lines) if output_lines else "")
            raise WorkerError(message + ("\n" + hint if hint else ""), error.get("phase"), output_lines, details)
        if result is None or code != 0:
            hint = hint_for("", output_lines)
            raise WorkerError(_crash_text(code, last) + ("\n" + hint if hint else ""), (last or {}).get("phase"),
                              output_lines, "\n".join(output_lines))
        finished = True
    finally:
        if not finished:
            _clean(output, existed)
    if hello is not None:
        result.setdefault("versions", hello.get("versions", {}))
    return result
