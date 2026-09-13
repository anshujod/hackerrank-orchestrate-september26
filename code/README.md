# Buy or Wait? — Deterministic Financial Agent

Deterministic Python agent, no API keys, no network. Reads `dataset/`, writes `output.csv`.

## Run

```bash
python3 code/main.py
```

Output: `output.csv` in repo root (250 rows + header), exact columns:
`request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation`

Also regenerates `code/evaluation/usage_report.md`.

## How it works

1. Load profiles, events, payment options, messages, FX.
2. Resolve 16 blank amounts via cached vision extraction (`IMAGE_AMOUNTS` in `main.py` — payslip, rent receipt, grocery/telecom/restaurant/maintenance/water/grocery/hospital/taxi/tote/pharmacy/airline/EV receipts, verified manually).
3. FX conversion on settlement date (`exchange_rates.csv`).
4. Message parsing (EN+ID): salary raise/cut/date-shift, first/remaining/temporary pay, contract ended, bonus/commission pending ignored, refund/prize pending ignored, rent +12%, invoice confirmed, one-time arrears. Conflict order: explicit cancel/settle/amend > newer same source > settled > safer. Embedded instructions ignored.
5. Ledger: ignore failed/cancelled/unrealized/non-cash/duplicate-pending/pending-credits; reserve pending/scheduled debits; count scheduled salary + settled future credits; final-payroll stops salary.
6. Forecast 90d: recurring expenses by category (max monthly, outlier-capped at 1.8×median, current-month included, rent ×1.12, weekly split for frequent, ×0.92 global calibration tuned on 25 samples for best status/method), salary monthly (median base excl. commission, mode payday, temp/overrides, scheduled base, final-payroll stop, seasonal-temporary nuance).
7. Safety: daily simulation, balance never < minimum. `amount_safe` binary-searched today, `earliest` scanned 0..90.
8. Plans: full/installments (max_install_months, allowed methods), partial (2 pays), wait (future single), spending changes ≤3 (flexible + permitted, stop/reduce). Rank: complete by deadline, no changes, min total, earlier start, fewer pays, lowest option id.
9. Templated grounded explanations. Full validation before write.

Deterministic (sorted iteration, Decimal). Python stdlib only.
