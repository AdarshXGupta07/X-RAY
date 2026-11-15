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
