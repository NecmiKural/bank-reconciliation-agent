# Agent log

Notable mistakes a coding agent made in this repository, and how they were caught. The point is not that the agent errs; it is that review, tests and measurement catch it before it reaches a reported number.

## 2026-10-18: holiday shift moved 215 of 250 payments onto public holidays

- **Proposed:** `shift_some_payments_onto_holidays` in `src/recon/synth.py` was meant to force 8 payments onto NZ public holidays. It stopped only when the count of holiday payments was exactly 8, but random payment dates already put about 10 lines on holidays, so the stop condition never fired and the loop moved every eligible line.
- **How it was caught:** all 19 tests passed on the first run, which was suspicious for a 500-line generator. Printing the distribution showed 215 of 250 lines in business A paid on a holiday, with 72 lines posting on a single day.
- **What changed:** the function now samples exactly 8 eligible rows. A new test bounds the share of holiday payments and the busiest posting day, so a regression fails CI. Lesson kept in AGENTS.md terms: a green test suite is evidence about what the tests check, nothing more; look at the data.

## 2026-10-18: green tests, ambiguous labels

- **Proposed:** the first generator drew partial-payment and fee amounts independently of the rest of the ledger, numbered invoices in the order the cases were built, and placed injections on lines that already had a unique exact amount.
- **How it was caught:** an independent review with three lenses (label validity, determinism and realism, security and rules), each finding checked by a second reviewer who tried to refute it. Confirmed: 6 of 15 held-out partial lines had more than one plausible invoice; one partial line also fitted two other invoices as a fee deduction; one split total had a second exact combination; invoice numbers, amount rounding and generic references revealed the case; 17 of 40 attacks named invoices issued after the payment, so a date filter alone defeated them; the held-out business reused the development business's PII and memo sentences.
- **What changed:** generation now runs in two phases, issuing every invoice before drawing line amounts and rejecting any draw that admits a second answer. Invoice numbers follow issue dates, amounts carry cents, attacks sit on lines a rules pass cannot close and name only invoices that existed at posting time, and the two businesses share no memo sentence. `tests/test_synth.py` now carries an oracle written apart from the generator, plus a fingerprint of the ground truth so any change to it is deliberate.
