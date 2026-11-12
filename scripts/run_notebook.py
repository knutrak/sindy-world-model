"""Execute a percent-format .py script ("# %%" cells) and save it as an .ipynb with outputs.

Usage: python scripts/run_notebook.py source.py|source.ipynb out.ipynb [working_dir]
An .ipynb source is re-executed (its old outputs are replaced).
Uses jupyter_client only (nbconvert / nbclient are not installed).
"""

import json
import re
import sys
import time
from queue import Empty

from jupyter_client.manager import start_new_kernel


def parse(src: str) -> list[dict]:
    parts = re.split(r"^# %%( \[markdown\])?[^\n]*\n", src, flags=re.M)
    cells = []
    for flag, body in zip(parts[1::2], parts[2::2], strict=True):
        body = body.strip("\n")
        if not body:
            continue
        if flag:
            lines = [
                ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                for ln in body.splitlines()
            ]
            cells.append(
                {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines).splitlines(True)}
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": body.splitlines(True),
                }
            )
    return cells


def _collect(kc, msg_id, n, timeout):
    outs, ok = [], True
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except Empty as err:
            raise RuntimeError("cell timed out") from err
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind, c = msg["msg_type"], msg["content"]
        if kind == "status" and c["execution_state"] == "idle":
            return outs, ok
        if kind == "stream":
            if outs and outs[-1]["output_type"] == "stream" and outs[-1]["name"] == c["name"]:
                outs[-1]["text"] += c["text"]
            else:
                outs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
        elif kind in ("execute_result", "display_data"):
            out = {"output_type": kind, "data": c["data"], "metadata": c.get("metadata", {})}
            if kind == "execute_result":
                out["execution_count"] = n
            outs.append(out)
        elif kind == "error":
            outs.append(
                {
                    "output_type": "error",
                    "ename": c["ename"],
                    "evalue": c["evalue"],
                    "traceback": c["traceback"],
                }
            )
            ok = False


def run(cells: list[dict], cwd: str, timeout: int = 3600) -> bool:
    km, kc = start_new_kernel(kernel_name="python3", cwd=cwd)
    n, ok = 0, True
    try:
        for cell in (c for c in cells if c["cell_type"] == "code"):
            n += 1
            start = time.time()
            outs, ok = _collect(kc, kc.execute("".join(cell["source"])), n, timeout)
            for out in outs:
                if out["output_type"] == "stream":
                    out["text"] = out["text"].splitlines(True)
            cell["outputs"], cell["execution_count"] = outs, n
            print(
                f"cell {n}: {time.time() - start:.0f}s"
                + ("" if ok else "  ERROR: " + outs[-1]["evalue"][:300])
            )
            if not ok:
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    return ok


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    cwd = sys.argv[3] if len(sys.argv) > 3 else "."
    if src.endswith(".ipynb"):
        with open(src) as f:
            cells = json.load(f)["cells"]
        for cell in cells:
            cell.pop("id", None)
    else:
        cells = parse(open(src).read())
    ok = run(cells, cwd)
    for k, cell in enumerate(cells):
        cell["id"] = str(k)
    meta = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(out, "w") as f:
        json.dump(
            {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5},
            f,
            indent=1,
            ensure_ascii=False,
        )
        f.write("\n")
    print("OK" if ok else "FAILED", out)
    sys.exit(0 if ok else 1)
