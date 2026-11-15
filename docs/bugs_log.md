# Bug Log

Log every problem that costs more than 20 minutes. These become interview answers.
Template: **Date · Symptom · Cause · How it was found · Fix · How to catch it faster next time.**

---

## 2026-11-15 — `python` command not found on Windows

- **Symptom.** Running `python --version` prints "Python was not found; run without arguments to install from the Microsoft Store".
- **Cause.** Python was installed through the `py` launcher and is not on PATH. Windows ships a placeholder `python.exe` that only points at the Store.
- **How it was found.** `py -0` lists the installed versions (3.14 and 3.13).
- **Fix.** Create the environment with `py -3.13 -m venv .venv`, then use the environment's own `python` after activating it.
- **Next time.** On Windows, check `py -0` first. Inside an activated environment `python` always works.

## 2026-11-15 — Choosing the PyTorch build

- **Symptom.** The default Python (3.14) is newer than what GPU libraries are usually first released for, and a plain `pip install torch` on Windows installs a CPU-only build.
- **Cause.** GPU builds of PyTorch live on PyTorch's own package index, one index per CUDA version.
- **How it was found.** `pip install --dry-run torch --index-url https://download.pytorch.org/whl/<cuda>` shows which version would be installed without downloading it.
- **Fix.** Python 3.13 environment; install from the `cu126` index before installing this package.
- **Next time.** After any PyTorch install, run a one-line check that the GPU is visible before doing anything else.
