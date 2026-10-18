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

## 2026-10-18: attacks placed where the model would never read them

- **Proposed:** the generator placed some injections on typo and duplicate-amount lines without checking whether a same-amount twin existed when the line posted. Separately, a database test asserted that the rules pass leaves every duplicate-amount line open.
- **How it was caught:** that test failed. Inspection showed the rules were right: when the twin invoice is issued after the payment posts, there is only one candidate, so the line is unambiguous. The real defect was elsewhere: 4 attacked lines per business were among the lines the rules close, so those attacks would never reach the model and the attack success rate would be measured on fewer cases than reported.
- **What changed:** attacks on typo and duplicate-amount lines now require a twin at posting time, and the test asserts the property that matters: no attacked line is closed by the rules pass, while the rules make zero wrong matches.

## 2026-10-18: a documented API rule that did not reproduce

- **Proposed:** the design notes, drafted with planning agents from the provider documentation, said that omitting `reasoning_content` from an earlier assistant turn makes DeepSeek answer 400 when tools are in use, and the test plan asked to observe that 400 once.
- **How it was caught:** a live check on `deepseek-flash` (thinking on, effort low, strict tools on the beta endpoint) accepted the stripped request and answered normally.
- **What changed:** nothing in the loop. It still appends every assistant turn unchanged, which the documentation requires and which costs nothing. The design note now says the rule is documented but was not enforced when tested, rather than stating it as fact.

## 2026-10-18: the verifier trusted any payer

- **Proposed:** `propose_match` checked amounts, one payer per match, overpayments and money out, but never whether the invoice belonged to the payer on the bank line. Payer similarity was computed against the name the model typed into the search, so it could not have served as a guard. A part payment counted as named if any token's digits matched the invoice number.
- **How it was caught:** an independent code review tried every retrieved invoice against business B. The verifier accepted 36 wrong single-invoice proposals, including a redirect attack naming another payer's invoice, so the prompt was the only defence. The same review found personal-data formats that escaped redaction (spaced IRD and bank numbers, landlines, `+64 (0)` forms) and showed that a bad API key would turn every line into a quiet escalation and a clean-looking run.
- **What changed:** similarity is computed against the bank line's own payee, and the verifier rejects a payer below 0.3 unless the reference field names the invoice; a database test checks that every labelled answer clears that bar. A part payment must name its invoice as a whole token. Redaction covers the extra formats and also runs on rationales. Only transient provider errors escalate a line; errors that would repeat on every line stop the run.

## 2026-10-18: metrics that said more than they proved

- **Proposed:** the first evaluation module reported the touchless rate and the wrong auto-approval bound over all lines, pooled attack success across all four goals and across three repeats as if every line-run were a new case, let any earlier match, right or wrong, make a payer "seen", and used a policy keyword screen that contained words from the held-out business's payloads.
- **How it was caught:** an independent review, with probes. 56 of 59 auto-approvals in business A came from the rules pass, not the agent. Preapproval and personal-data attacks are decided by the policy screen and the redactor, so the model could not move them. The pooled bound looked about five times tighter than the 20 distinct model-decided attack lines support. Words such as "correction" and "sign-off" occurred only in business B's payloads. The same review found that approving a part payment marked the invoice paid, that bank-rule auto-approvals had no sign or size limit, and that an unanchored `.gitignore` entry would have kept the CI recordings out of the repository.
- **What changed:** every rate is reported on the part of the pipeline that produced it. Model-decided and code-enforced attacks are reported apart, and bounds over repeats count distinct lines. A payer becomes seen only after a correct match. The screen's words now come from business A by a stated rule, and the results report how it does on B, false positives included. Part payments leave the invoice open and allocations carry amounts. Bank rules have envelopes. The gate compares rules decisions and policy verdicts as well as agent decisions, and provider errors invalidate a run.

## 2026-10-18: results recorded while the code was changing

- **Proposed:** the first round of defence ablations ran in the background while the report module was being written in the same checkout, and its results were about to be published.
- **How it was caught:** the `commit` field that every results file records said `-dirty` for five of the six files. The changed files were reporting code that the runs never call, but nothing in the results could show that to a reader.
- **What changed:** the reporting code was committed first and all six ablation runs were repeated on the clean commit. A long run now goes on a committed tree with nothing edited under `src/` until it finishes.

## 2026-10-18: a README that claimed more than the code does

- **Proposed:** the first full README said the model never does arithmetic, that the audit table was append-only even for the owner, that the verifier refuses another payer's invoice, and that the agent's database role could not post; it called the second business held out, reported a pre-approval result that could only be zero, and set one-run ablations beside three-run results without saying so. `make demo`, the command it promised works offline, crashed after replaying every line.
- **How it was caught:** a claims audit by a separate review agent that checked each sentence against the code, the tests and the results files, and ran every command the README gives.
- **What changed:** each claim now says what the mechanism actually is: no model arithmetic is trusted; the trigger stops updates, deletes and truncation but not a superuser; the payer check is a name-similarity threshold; the role is switched per transaction from an owner connection. Business B is described by how it was and was not used, the zero-by-construction row is gone and named as a gap, and the ablation is labelled as two groups in single runs. `recon run` is fixed and covered by a test that replays the slice end to end.
