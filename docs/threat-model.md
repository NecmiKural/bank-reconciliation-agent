# Threat model

What can go wrong when a language model helps reconcile bank lines, and which control in this repository stands in the way. Categories follow the OWASP Top 10 for LLM Applications (2025).

## What is worth protecting

- **The books.** A wrong allocation hides an unpaid invoice or pays the wrong customer's balance.
- **Personal data** in payer-written text: IRD numbers, bank accounts, phone numbers, emails.
- **The audit trail**, which shows who approved what.
- **Spend** on the model provider.

## Who controls what

| Input | Written by | Trusted |
|---|---|---|
| Amount, posting date | the bank | yes |
| Payee name | the payer's bank, from the account name | partly |
| Reference (12 characters) and memo | the payer | no |
| Invoices and contacts | the business | yes |
| Model output | the model | no: checked before it counts |

## Threats and controls

| OWASP LLM 2025 | Threat here | Control | Test or measurement |
|---|---|---|---|
| LLM01 Prompt injection | A memo tells the agent to match another payer's invoice, settle more invoices than the payment covers, skip review, or repeat personal data | Payer text fenced as data in the prompt; verifier refuses another payer's invoice, sums that do not add up and invoices the search never returned; no posting tool; shadow policy keeps instruction-like text from auto-approval; human review | `tests/test_tools.py`, `tests/test_policy.py`; attack success per goal and the defence ablation in the README |
| LLM02 Sensitive information disclosure | Personal data in memos reaches the provider or the queue | Redaction before every request and again on rationales, with IRD check-digit validation | `tests/test_redact.py` covers every planted value in both businesses; every outbound request is scanned during evaluation |
| LLM05 Improper output handling | Model text is trusted as a decision, or printed raw to a reviewer's terminal | Decisions exist only as verified tool calls; rationales are redacted; control characters are dropped before printing | `tests/test_agent.py`, `tests/test_tools.py` |
| LLM06 Excessive agency | The agent changes the books | No tool writes the ledger; the agent's Postgres role can only read the ledger and insert proposals; the audit table is append-only through a trigger | `tests/test_db.py` tries the forbidden writes and reads |
| LLM07 System prompt leakage | The prompt reveals something exploitable | The prompt holds no secrets or credentials; the controls are in code, so knowing the prompt does not help an attacker | not applicable |
| LLM10 Unbounded consumption | A line loops or a provider fails over and over | Step, token and cost limits per line; errors that would repeat on every line stop the run; a prepaid provider balance caps total spend | `tests/test_agent.py` |
| LLM03 Supply chain | A compromised dependency or CI action | Locked dependencies (`uv.lock`), GitHub Actions pinned by commit SHA, no agent framework or model router | CI runs `uv sync --locked` |

## What remains

- **A payload worded as an ordinary remittance passes the keyword screen.** It still meets the verifier and, outside the narrow auto-approval case, a human.
- **A payer who really is the account holder can name the wrong invoice in the reference field.** The verifier accepts the payer through the reference; the policy still routes the line to review unless every auto-approval condition holds.
- **The model may be fooled on a line where the books allow the attacker's choice**, for example two invoices of the same payer for the same amount. That is a judgement error with a correct payer, and review is the control.
- **A local model on another machine is reached over the LAN without TLS.** The setup binds it to one host by firewall rule. It is not suitable for real data as configured.
