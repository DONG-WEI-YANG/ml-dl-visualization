# F12 dependency remediation — 2026-09-20

## Scope

Frontend dependencies were updated locally. Backend candidates were installed only in a temporary target directory; the shared `.venv` and deployed image were not upgraded. F12 is partially remediated, not fully closed.

## Frontend

Paired `vitest` and `@vitest/coverage-v8` at **4.1.11**, the patched maintained 4.x release in the [maintainer advisory](https://github.com/vitest-dev/vitest/security/advisories/GHSA-82fw-gwwq-j7x9). Registry metadata permits Vite 6 and Node 20/22/24. Local Node was 24.18.0; CI's Node 20 is supported. `npm audit fix` (without `--force`) also updated compatible humanfs and fflate dependencies. The resulting lockfile audit reports **0 vulnerabilities**, previously five moderate package entries. The first build passed with existing large-chunk warnings; final full regression is recorded below.

These are development-server/specific ZIP or filesystem API advisories, not evidence of static-site compromise: [fflate](https://github.com/advisories/GHSA-px8p-9vwx-vf98), [humanfs](https://github.com/advisories/GHSA-p498-v437-472g).

## Backend candidate pins

`requirements.txt` and `pyproject.toml` now declare the same runtime dependencies, including the previously omitted NLP requirements. Exact security-reviewed pins in both files:

| Package | Version |
|---|---|
| FastAPI | 0.141.1 |
| Starlette | 1.3.1 |
| AnyIO | 4.14.2 |
| Pydantic | 2.13.5 |
| pydantic-settings | 2.14.2 |
| python-multipart | 0.0.31 |
| NLTK | 3.10.3 |
| click | 8.5.0 |
| idna | 3.20 |

Versions/Python requirements were checked via primary PyPI JSON metadata, including [FastAPI](https://pypi.org/pypi/fastapi/0.141.1/json) and [Starlette](https://pypi.org/pypi/starlette/1.3.1/json). Selected packages support Python 3.11; FastAPI permits this Starlette version. This is a pinned tested subset, **not a complete cross-platform lockfile**.

Primary advisories reviewed: [Starlette URL reconstruction](https://github.com/Kludex/starlette/security/advisories/GHSA-jp82-jpqv-5vv3), [Starlette form limits](https://github.com/Kludex/starlette/security/advisories/GHSA-82w8-qh3p-5jfq), [multipart DoS](https://github.com/Kludex/python-multipart/security/advisories/GHSA-mj87-hwqh-73pj), [AnyIO process-pool stderr](https://github.com/agronholm/anyio/security/advisories/GHSA-5p39-cfhj-2xmp), [settings nested secrets](https://github.com/pydantic/pydantic-settings/security/advisories/GHSA-4xgf-cpjx-pc3j).

Reachability differs: Starlette handles HTTP and curriculum FileResponse; source review found no Form/UploadFile endpoints or StaticFiles mounts. Settings use environment/.env rather than NestedSecretsSettingsSource; models use AnyIO threads rather than process pools. These observations prioritize work, not erase advisories.

## Backend validation

Installed candidates with `pip install --target C:/Users/user/AppData/Local/Temp/ml-dl-security-stack-20260920`, retaining existing venv packages for other dependencies. With that target first on PYTHONPATH:

- Full backend suite: **229 passed**, three warnings (existing SnowNLP warnings plus Starlette recommending httpx2 for TestClient).
- `pip-audit --path <target>`: **one known NLTK vulnerability**, no advisories for the other 20 target packages.
- Local platform: Python 3.14/Windows. This does not replace clean Linux/Python 3.11 image build, model downloads, or deployed-image audit.

## Explicit residuals

1. **NLTK GHSA-8mgp-746c-j5xp remains unpatched**, including 3.10.3, per [upstream](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp). It affects caller-controlled model artifact paths bypassing pathsec. Application usage found sentence tokenization and fixed English stopwords, not the affected persistence APIs/user-selected model paths. Keep tracked without an audit ignore and reassess future model persistence features.
2. The original venv remains unchanged. Original findings in aiohttp, datasets, msgpack, Pillow, pip, Pygments, pytest, requests, setuptools, torch, transformers and urllib3 were **not remediated or fully exploitability-classified**. IDs/versions remain in `audit-2026-09-20-python-dependencies.json`. Optional NLP/embedding/model-loading paths need separate regression and image inventory. No torch binaries or blanket upgrades were attempted.
3. Clean installs may resolve unpinned dependencies differently. Before deployment, build the clean production image, run regression and pip check, capture its complete inventory/lock and audit that image. Existing-environment tests do not validate arbitrary future resolution.
4. No production package inventory was accessed and no deployment occurred.

## Final verification

- Frontend after concurrent application fixes: **23 files / 134 tests passed** on Vitest 4.1.11; lint clean; build passed (existing >500 KB chunk warnings).
- Final `npm audit --json`: zero vulnerabilities. Resolved fixes include humanfs 0.16.8 and fflate 0.6.11 / 0.8.3; Vite remains 6.4.3.
- `pip install --dry-run -r requirements.txt --report <temporary-report.json>` completed successfully against the current venv. It proposed 11 package changes and no torch download/install. This is dependency resolution, not a clean-environment or production verification.
- A parsed comparison confirms requirements.txt and pyproject.toml agree on all **38 runtime dependencies**.

## Follow-up: networking/image candidates and runtime exposure

The following additional exact pins are now in both manifests: **aiohttp 3.14.3, Pillow 12.3.0, Requests 2.33.0, urllib3 2.7.0**. They were installed into a second isolated target (`ml-dl-security-network-20260920`). Combined with the web/NLP target, the expanded backend suite passed **236 tests**, three warnings. Direct smoke checks passed for aiohttp session creation, Pillow in-memory PNG encode/decode, Requests request preparation, and urllib3 URL parsing. Audit of all 14 packages in the network/image target reported **zero vulnerabilities**. These smoke checks do not exercise all optional ML/network paths.

The earlier residual list is narrowed by these additional candidates; the shared venv still has its original versions. Runtime classification below is based on explicit application callsites and the current advisory descriptions, not a blanket exploitability assurance.

| Component | Application callsites / exposure | Status / next action |
|---|---|---|
| aiohttp | No direct app import/server. Indirect dependency through datasets/fsspec from pycorrector; app/nlp/text_correction.py:18 uses ProperCorrector. Actual Wikipedia enrichment uses httpx at app/rag/web_enricher.py:194 and :221. | Patched candidate 3.14.3 tested/audited. The application does not expose an aiohttp server; indirect client/model download behavior still belongs in clean-image tests. |
| Pillow | No application Image.open/PIL import or image upload route. Installed through plotting/image/ML dependencies; matplotlib and scikit-image are declared. | Patched candidate 12.3.0 tested/audited. No user-image decoder path found in current API. |
| Requests / urllib3 | No direct application imports. scripts/ingest_to_cloud.py:22 uses Requests for operator-selected backend URLs and fixed API routes; also transitives of ML/download libraries. | Patched candidates 2.33.0 / 2.7.0 tested/audited. Requests advisory requires direct extract_zipped_paths use, absent in app/scripts. urllib3 streaming/decompression issues remain relevant to dependency consumers of remote responses, motivating patch. |
| transformers / torch | Active optional semantic inference: app/nlp/context_memory.py:25 and app/nlp/retrieval.py:23 instantiate SentenceTransformer with the fixed paraphrase-multilingual-MiniLM-L6-v2 model. Text is user controlled, model ID/checkpoint path is not. | **Residual**: installed transformers 5.3.0 and torch 2.10.0 remain. Audit patch targets are transformers >=5.10.0 and torch >=2.13.0. No application LightGlue, save_pretrained, or torch.jit.script call found. Full actual-model encode regression is required before upgrading these tightly coupled heavy dependencies. |
| datasets | Transitive pycorrector installation; no app load_dataset or user-specified dataset path. | **Residual** installed 4.7.0, recorded fix 5.0.1. Optional library internals not exhaustively traced; verify pycorrector/model-data behavior with clean image before updating. |
| msgpack | Transitive wordfreq; app/nlp/difficulty.py:87 calls word_frequency using packaged linguistic data. No app msgpack decoder or user binary input. | **Residual** installed 1.1.2, recorded fix 1.2.1. Test wordfreq behavior on candidate dependency before pinning. |
| Pygments | No direct application use; transitive rich/typer tooling. | **Residual**, local tooling/transitive scope; no syntax-highlighting endpoint found. |
| pip / setuptools / pytest | Installer/build/test tools, not application request handlers. | **Residual** developer/image build supply-chain scope; update separately with build/test validation. No claim these are present in deployed image at scanned versions. |

Source search found no FastAPI Form/File/UploadFile handlers and no API for selecting semantic model repositories, checkpoints or arbitrary image files. Fixed model IDs still leave model supply-chain/revision risks: the SentenceTransformer calls do not pin a model revision, so this is a remaining reproducibility concern, not evidence that all model artifacts are safe.

Primary upstream evidence: [Requests affected utility and fix](https://github.com/psf/requests/security/advisories/GHSA-gc5v-m9x4-r6x2), [urllib3 streaming decompression](https://github.com/urllib3/urllib3/security/advisories/GHSA-mf9v-mfxr-j63j), [Transformers LightGlue fix](https://github.com/huggingface/transformers/commit/676559d5022b74aaa0cee1cee0842b7f27c5320e), [Transformers chat-template save fix](https://github.com/huggingface/transformers/commit/eaaaf8494dd5386634ae37d1d122212fdc315be5), [PyTorch JIT annotation fix](https://github.com/pytorch/pytorch/commit/b90c94991cdf8b87c8f7439f79518e0ef2c4ca4f). No large torch download, shared-venv update, external API POST, or deployment was performed.

Final follow-up validation: the expanded requirements dry-run resolves successfully against the existing venv; manifests agree on 42 runtime dependencies. No actual venv install occurred.

Root integration verification after the final stale-verification regression: current venv backend 236 passed / 0 skipped; frontend 135 passed, lint clean, build passed, npm audit zero. These supersede the earlier intermediate frontend 134 count; isolated combined candidate backend remains 236 passed.
