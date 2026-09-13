# Usage Report — Buy or Wait? final full-dataset run

Run date (UTC): 2026-09-13T05:09:20.783945+00:00
Requests evaluated: 250

## Approach
Deterministic Python engine (`code/main.py`) for ledger, FX, 90-day safety simulation, plan ranking. No per-request LLM calls in final run. Vision/OCR for 16 blank-amount images done once offline (cached in `IMAGE_AMOUNTS`), reused deterministically.

## Models
| Provider | Model | Purpose | Calls | Input tokens (est) | Output tokens (est) |
|---|---|---|---|---|---|
| Manual/VLM inspection (opencode+Muse Spark vision) | image-amount extraction | 16 blank amounts (event_253 etc.) | 16 | ~24000 (~1500/img) | ~1600 (~100/img) |
| None (deterministic) | n/a | main 250-request run (forecast+simulate+rank) | 0 | 0 | 0 |

## Totals (final run producing output.csv)
- Total model calls: 0 (main) + 16 offline vision (cached, not repeated)
- Total tokens (main run): 0 input + 0 output = 0
- Average per request (main run): 0 input, 0 output
- Offline vision (one-time): ~24000 input, ~1600 output, ~25600 total (~1600/req avg if amortized over 250, ~1600/img)

## Cost (estimated)
- Main run: $0.00 total, $0.00 per request (no API).
- Offline vision if via API (e.g., GPT-4o-mini $0.15/1M in, $0.60/1M out): ~$0.0036 + $0.00096 ≈ $0.005 total, ~$0.00002/req amortized. Actual dev used built-in vision at no billed API cost.

## Repro
`python3 code/main.py` reads `dataset/` and writes `output.csv` deterministically (no network, no API keys).
