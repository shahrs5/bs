# Changelog

All notable benchmark, data, and viewer changes are tracked in this file.

## [2.0.1] - 2026-03-04

### Added
- Added benchmark runs for `openai/gpt-5.3-chat` and `google/gemini-3.1-flash-lite-preview`.

### Changed
- Set launch date metadata for both models to `2026-03-04` and synced to:
  - `data/model_metadata/model_launch_dates.csv`
  - `data/latest/model_launch_dates.csv`
  - `data/v2/latest/model_launch_dates.csv`
- Updated `data/latest/leaderboard_with_launch.csv` and `data/v2/latest/leaderboard_with_launch.csv` to show the new launch date and model age (`0`) for both models.
- Updated [viewer/index.v2.html](viewer/index.v2.html) launch metadata loading to merge embedded rows with CSV rows and fetch metadata with `cache: "no-store"` so new model dates reliably appear in all launch charts.

## [2.0.0] - 2026-03-01

### Highlights
- Added `100` new v2 nonsense questions.
- Added domain-specific coverage across `5` domains: `software` (40), `finance` (15), `legal` (15), `medical` (15), `physics` (15).
- Added new v2 visualizations (model detection mix, domain landscape, over-time trends, release-date scatter, and reasoning tokens/cost scatter).

### Added
- New v2 question set in [questions.v2.json](questions.v2.json) with 100 prompts across 5 domain groups and 13 techniques.
- New v2 config in [config.v2.json](config.v2.json) with high-throughput collection defaults and updated technique set.
- Dedicated v2 viewer page at [viewer/index.v2.html](viewer/index.v2.html).
- Dedicated published v2 dataset in `data/v2/latest/*`.
- Question-builder script [scripts/build_questions_v2_from_draft.py](scripts/build_questions_v2_from_draft.py).

### Changed
- Viewer and docs now support side-by-side versioning (`v1` and `v2`) without overwriting older data.
- Pipeline/docs updated for explicit v2 publishing via `--config config.v2.json --viewer-output-dir data/v2/latest`.
- Publish pipeline now scrubs local machine path fragments from published JSONL artifacts.
- Canonical panel policy is now fixed to exactly three judges (`panel_mode=full`) with `mean` aggregation in the main pipeline.
- Viewer categorying now uses published `status` + `consensus_score` as canonical defaults (when all judges are selected), while still allowing subset-judge exploratory views.
- `viewer/index.v2.html` CSV parsing is now quote-aware for launch metadata and other future CSV extensions.
- `viewer/index.v2.html` now includes friendly labels for legacy v1 technique keys.

### Removed / Cleaned
- Removed obsolete `v2_old` drafts and local run-history artifacts.
- Removed local-only temporary/debug files before publish.

## [1.0.0] - 2026-02-25

### Added
- Initial public benchmark release (v1 dataset + viewer).
