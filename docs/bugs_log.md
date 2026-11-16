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

## 2026-11-15 — A docs update emptied PROGRESS.md

- **Symptom.** A Python script that edited `docs/PROGRESS.md` crashed with `UnicodeEncodeError: 'charmap' codec can't encode character '≈'`, and the file was left at 0 bytes.
- **Cause.** On Windows, `Path.read_text()` / `write_text()` and `open()` default to the system code page (cp1252), not UTF-8. The text contained "≈", which cp1252 cannot encode. `write_text` had already truncated the file before the encode failed.
- **How it was found.** `wc -c` showed 0 bytes; an `iconv` scan showed every other file was still valid UTF-8.
- **Fix.** Restored the file with `git checkout -- docs/PROGRESS.md`, then redid the edit with `encoding="utf-8"`.
- **Next time.** Always pass `encoding="utf-8"` when reading or writing text in Python on Windows (or set `PYTHONUTF8=1`). Commit before bulk edits so `git checkout` can undo them.

## 2026-11-15 — Windows blocked a scikit-learn file

- **Symptom.** `pytest` failed to import `sklearn.linear_model` with `ImportError: DLL load failed while importing _libsvm_sparse: An Application Control policy has blocked this file.`
- **Cause.** A Windows application-control feature (this message is typical of Smart App Control or a similar policy) checked a newly installed compiled file and blocked it on first load.
- **How it was found.** Importing each scikit-learn submodule on its own a minute later: all loaded. Two further test runs passed.
- **Fix.** None needed; it did not come back. Security settings were not changed.
- **Next time.** If the error recurs, it is a Windows security decision, not a Python bug. Options are yours: check Windows Security → App & browser control, or replace the blocked piece (here, logistic regression) with NumPy/SciPy code.

## 2026-11-15 — A "successful" download that was a web page

- **Symptom.** The Shenzhen zip download finished instantly with exit code 0. The 3.6 GB "zip" was 13,854 bytes.
- **Cause.** The server answered with HTTP 200 and an HTML error page instead of the file. `curl --fail` only catches 4xx/5xx status codes, so it reported success.
- **How it was found.** The file size, then `file` reported "HTML document".
- **Fix.** Downloaded from NLM's official per-file folders with a script that rejects HTML responses and checks every file's size against the listing.
- **Next time.** After any download, check size and type (or a checksum) before trusting it. Exit code 0 is not proof.

## 2026-11-16 — Clinical readings parser missed real formats

- **Symptom.** 9, then 20 Shenzhen rows had no age or sex.
- **Cause.** The readings use at least five layouts: "male 35yrs", "femal 32yrs" (typo), "female24yrs" (no space), "Female, 28yrs" (comma) and infant ages in months or days. A rewrite fixed the first ones and broke the comma rows.
- **How it was found.** Printing the rows with missing values and their raw bytes (`od -c`).
- **Fix.** One pattern for the Shenzhen line (`sex[\s,]*number unit`), unit conversion to years, and a test case for every real format found.
- **Next time.** After a parser change, count missing values before and after; a drop in one place can hide a rise in another. Put each real-world example in a test.

## 2026-11-16 — Smart App Control blocks SciPy and scikit-learn at random

- **Symptom.** Imports failed at random with `ImportError: DLL load failed ... An Application Control policy has blocked this file`, each time on a different file (`_ckdtree`, `_highs_options`, `_superlu`, `_libsvm_sparse`). Three runs in a row failed.
- **Cause.** The Windows Code Integrity log (event IDs 3033/3077) shows the files "did not meet the Enterprise signing level requirements". Smart App Control is on (registry state 1). It blocks unsigned compiled files that Microsoft has no reputation data for. The newest pip wheels are the least known.
- **How it was found.** Reading `Microsoft-Windows-CodeIntegrity/Operational` with `Get-WinEvent`, then comparing: Anaconda's own SciPy 1.16.3 and scikit-learn 1.7.2 imported 5 out of 5 times.
- **Fix.** The security setting was left as it is. The project environment now uses Anaconda's NumPy, SciPy, scikit-learn, pandas, matplotlib, Pillow and PyYAML (`include-system-site-packages = true`, pip copies uninstalled); only PyTorch comes from pip. The full test suite then passed repeatedly.
- **Next time.** On Windows, check the Code Integrity log as soon as an "Application Control" error appears. Prefer long-established builds over brand-new wheels.

## 2026-11-16 — Two OpenMP runtimes: worker crash and "OMP Error #15"

- **Symptom.** `DataLoader` workers died with `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`. Later the main process crashed the same way when PyTorch was imported before NumPy.
- **Cause.** Anaconda's NumPy (MKL) and PyTorch each ship their own `libiomp5md.dll`. `psutil` showed both loaded in one process even when the program happened not to crash.
- **How it was found.** Listing the process's loaded files with `psutil.Process().memory_maps()` and filtering for `iomp5`.
- **Fix.** Renamed `torch\lib\libiomp5md.dll` to `.bak`. PyTorch then loads Anaconda's copy from `anaconda3\Library\bin`, so exactly one runtime is loaded in either import order. Training also uses `num_workers=0`. The `KMP_DUPLICATE_LIB_OK=TRUE` workaround was rejected: Intel warns it can silently give wrong results.
- **Next time.** When mixing conda and pip, count loaded copies of shared runtimes instead of assuming. "It didn't crash" does not mean only one copy is loaded.

## 2026-11-16 — CPU-only tests failing with "CUDA error: out of memory"

- **Symptom.** `test_training.py` failed now and then with `torch.AcceleratorError: CUDA error: out of memory`, although every test trains on the CPU. Hiding the GPU with `CUDA_VISIBLE_DEVICES=""` made it worse: an occasional hard crash (`Windows fatal exception: code 0xc0000374`) at the same line.
- **Cause.** AdamW's `step()` runs `_accelerator_graph_capture_health_check`, which asks the GPU for its current stream whenever PyTorch was built with CUDA. That opens a CUDA context. The 4 GB card is shared with Ollama and the browser, so sometimes there was no room for it.
- **How it was found.** The traceback pointed into `torch/optim/optimizer.py`; reading that function showed it runs regardless of the training device. Repeating the suite several times exposed the intermittency.
- **Fix.** `tests/conftest.py` switches that health check off for the test session only (it only matters for CUDA-graph capture, which no test uses). Six full runs in a row passed. Real training is unchanged.
- **Next time.** A failure that comes and goes usually depends on something outside the test (shared GPU, memory, timing). Run the suite several times before and after a fix, and keep unit tests independent of the GPU.

## 2026-11-16 — Training crashed with "not enough memory" (RAM, not GPU)

- **Symptom.** During the multi-seed runs: `DefaultCPUAllocator: not enough memory`, then `CUDA out of memory ... 2.33 GiB is free`, and earlier "The paging file is too small" when running tests at the same time as training.
- **Cause.** Windows ran out of commit memory (RAM plus page file): 0.1 GB of 27.9 GB free. The Ollama model server held about 6 GB, the browser about 2 GB, and each PyTorch process about 2–4 GB including the CUDA libraries. A "CUDA out of memory" error with plenty of free GPU memory was really a host-memory failure.
- **How it was found.** `Get-CimInstance Win32_OperatingSystem` (free physical and virtual memory) and the process list sorted by memory.
- **Fix.** Ran one training process at a time, with nothing else heavy alongside, and a retry. The runner now saves its summary after every model, so a crash loses only the model in progress. Seeds reproduced exactly on rerun (same AUCs), confirming the seeding.
- **Next time.** Before long GPU jobs, check free commit memory, not just GPU memory, and close other large programs (Ollama). Save progress incrementally in any long job.
