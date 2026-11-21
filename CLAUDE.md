# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read **AGENTS.md** first: it has the current version, the public model families, recommendation policy and doc
pointers. `docs/AI_REFERENCE.md` holds the source-backed facts (models, APIs, hardware limits); `docs/DETAILS.md`
covers how the engine works, the measured numbers and every setting.

## What this is

Strata runs large MoE models (Qwen3.8-Flash-Next 125B, Qwen3.6-35B-A3B; GGUF and other packs) on one or a few
consumer GPUs plus system RAM. Since the 0.2.16 layout move the top level is `engine/`, `terminal/`, `app/`, `docs/`,
`tools/` and `.github/`. It has three parts:

- **Engine (C++20 / CUDA, HIP, SYCL, Metal):** `engine/` holds what CMake builds: `engine/src/`,
  `engine/include/strata/` (layouts mirror each other), `engine/metal/`, `engine/sycl/`, `engine/tests/`. Each model
  family is its own executable, and they all speak the same `--serve` stdin/stdout protocol:
  - `strata` is the Qwen3.8 engine. `engine/src/program/generate.cpp` is its driver: embed, captured layer graphs,
    lm_head, sample.
  - `strata-q36` is Qwen3.6 (`engine/src/qwen35/`). `strata-q36ref` is its reference, checked against llama.cpp by
    `terminal/tools/qwen35/`.
  - `strata-vision` handles images (built from `terminal/tools/vision/`).
  - None of the engines tokenize. They take token ids.
- **Server (Python):** `terminal/serve/server.py` exposes the OpenAI API (`/v1/chat/completions`, Responses) and the
  Anthropic API (`/v1/messages`), plus a web page. Much of it moved unchanged into `terminal/serve/server_parts/`
  (`base.py`, `core.py`, `http_api.py`, `engines.py`, ...); `serve.server` imports those names back, but a test that
  patches a helper must patch the module that defines it. The API sits behind an `Engine.generate(...)` boundary:
  `StrataEngine` wraps the resident engine process, and `MockEngine` (`--engine mock`) is a stand-in, so every API
  path is testable without a GPU.
- **Ways to install and run it:**
  - **The Terminal** (`terminal/strata.py` → `terminal/strata_cli/` for commands and the TUI,
    `terminal/strata_control/` for the logic): the `strata` command (`strata launch|serve|connect|monitor|bench|tune
    ...`, wrappers in `terminal/bin/`). This is the release path for people and agents with a shell. `strata.py`
    stays dependency-free, and only `serve`/`bench` need `terminal/requirements.txt`.
  - **The app:** `app/desktop` (Tauri) and `app/host` (the Node host plus a Vite/TS web UI). Installers are built in
    `tools/packaging/`.
  - **Source setup:** `terminal/setup.py` (started by `START-HERE.bat` or `setup.sh` at the root). This is the
    developer and fallback path. It checks the PC, downloads and packs the model (`terminal/tools/strata_pack.py`
    ...), fetches a prebuilt engine from the signed feed or compiles one, writes `strata-<model>.json` and
    `run-<model>.bat`, then serves on :8080. `UPDATE.bat` (or `START-HERE.bat --update`) updates a source install.
    `terminal/tools/strata_bench.py` (`tools/BENCH.bat`) is the benchmark.

Memory tiers (`docs/HOW_IT_WORKS.md`, `docs/paper/`):
- The GPU holds the mixers, routers, shared experts, head, MTP draft layer, KV cache and an adaptive expert cache.
- Every expert sits pinned in RAM; experts not in the GPU cache are computed on the CPU (AVX-512/AVX2, ggml
  i-quants) at the same time.
- Low-RAM modes read experts from the GGUF on disk.
- Speculation: MTP drafts up to 3 tokens and prompt lookup up to 5, verified in one pass.

## Build

`terminal/setup.py`'s `cmake_build` is the reference, run inside vcvars on Windows. By hand:

```
cmake -G Ninja -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON [-DCMAKE_CUDA_ARCHITECTURES=120]
cmake --build engine/build --target strata          # or strata-q36
```

- **Backends:** `STRATA_ENABLE_CUDA` (default OFF; without it only the CPU-side artifact/plan tools build),
  `STRATA_ENABLE_HIP` (wave32 AMD), `STRATA_HIP_GFX906`, `STRATA_ENABLE_SYCL` (experimental Intel Arc).
- **Other options:** `STRATA_PORTABLE` (AVX2 baseline plus static MSVC runtime, used for releases),
  `STRATA_BUILD_TESTS`, and many kernel feature switches (`STRATA_MMQ_*`, `STRATA_Q6K_EXPERTS`, ...). See the
  `option(` lines in `engine/CMakeLists.txt` and `engine/cmake/`.
- **Guides:** AMD in `docs/AMD_HIP.md`; Mac in `docs/MAC.md` (`engine/metal/CMakeLists.txt`).

## Versions

- The engine version is `project(strata VERSION ...)` in `engine/CMakeLists.txt`.
- The app and Terminal version is in `app/host/package.json`, `AGENTS.md` and the release notes in
  `docs/release-notes-*.md`.
- `MIN_ENGINE` in `terminal/setup.py` makes source installs replace older engines. Raise it when an engine change
  must reach users.
- `terminal/engine/BUILD.json` records the installed engine: its version, `source: local|release`, and a source
  fingerprint. Setup rebuilds a local engine when the fingerprint changes.

## Tests

Use the repo venv (`terminal/.venv/Scripts/python.exe` on Windows, which START-HERE.bat makes; the system Python
lacks `regex` etc.). None of these need a GPU. `terminal/conftest.py` points APPDATA/XDG/LOCALAPPDATA at a temp dir,
so pytest runs never touch the real user config or state. Run the Python suites from `terminal/`:

- Server (as CI runs it): `python -m unittest discover -s serve -p "test_*.py"`. One file or test:
  `python -m unittest serve.test_server.ListenProblem -v`.
- Terminal and Control: `python -m pytest tests/terminal` (`-k name` for one). pytest is not in `requirements.txt`,
  so `pip install pytest` into the venv first.
- Setup and tools: `python -m unittest discover -s tools -p "test_*.py"`, or one file directly:
  `python tools/test_setup_<name>.py`. The repository's own checks (links, the results index, ...) are in the
  top-level `tools/` and run from the root: `python tools/check_links.py`, `python tools/bench_results_index.py --check`.
- App UI: `npm test` in `app/host`. `npm run check` runs typecheck plus the UI audits.
- C++: parity and unit tests register with CTest, most run as `<name> --selftest`. Run them with
  `ctest --test-dir engine/build [-R name]`. They live beside the code (`*_parity.cpp`, `*_test.cpp`) and under
  `engine/tests/` (per area: core, cuda, hip, sycl, prefill, program, ...).
- Each Python suite takes minutes. Run the narrowest one that covers the change.

## Contributing rules (.github/CONTRIBUTING.md, enforced upstream)

- **Titles:** every issue and PR title starts with an area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`. A CI check fails a PR without one.
- **Scope:**
  - One problem per PR.
  - New features are opt-in, and the default path stays byte-identical to the last release.
  - Agree a large feature, refactor or default change in an issue first.
- **PR body:** problem and result, then the approach, then validation with **passed / failed / not run** kept
  apart, then the limits.
- **Evidence:**
  - Build every backend a change touches.
  - A default changes only with before/after measurements on named hardware.
  - Never present invented results. Mark a finding from reading code as "unconfirmed static analysis".
- **Docs style:** plain words, short sentences, non-technical readers first, and every number with what it was
  measured on.
- **Code comments:** long-form, explaining *why*, often citing issue/PR numbers (`#1234`) and plan phases. Match
  that style.
- **Benchmarks:** `docs/bench-results/<date>-<topic>/` (listed in its index, `tools/bench_results_index.py`) and
  `docs/benchmarks/` hold the measurements behind performance claims.
