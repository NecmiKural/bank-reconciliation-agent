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
| LLM01 Prompt injection | A memo tells the agent to match another payer's invoice, settle more invoices than the payment covers, skip review, or repeat personal data | Payer text fenced as data in the prompt; verifier refuses an invoice whose payer name is not similar to the bank payee unless the reference names it, sums that do not add up and invoices the search never returned; no posting tool; shadow policy keeps instruction-like text from auto-approval; human review | `tests/test_tools.py`, `tests/test_policy.py`; redirect and over-allocation success and the two-group defence ablation in the README |
| LLM02 Sensitive information disclosure | Personal data in memos reaches the provider or the queue | Redaction before every request and again on rationales, with IRD check-digit validation | `tests/test_redact.py` covers every planted value in both businesses; every outbound request is scanned during evaluation |
| LLM05 Improper output handling | Model text is trusted as a decision, or printed raw to a reviewer's terminal | Decisions exist only as verified tool calls; rationales are redacted; control characters are dropped before printing | `tests/test_agent.py`, `tests/test_tools.py`, `tests/test_gate.py` |
| LLM06 Excessive agency | The agent, or an outside assistant through the MCP server, changes the books | No tool writes the ledger; the agent's queries run under a Postgres role that can only read the ledger and insert proposals, switched to per transaction from the owner connection; a trigger rejects updates, deletes and truncation of the audit table; the MCP server exposes only read-only tools, marked as such, running under the same role | `tests/test_db.py` tries the forbidden writes and reads; `tests/test_mcp.py` lists the tools |
| LLM05 Improper output handling, LLM06 Excessive agency: code mode | Model-written code reads the network or the database, finds a key, writes files, exhausts the host, floods the context with output, forges a decision, or calls tools beyond its line | The code runs in a throwaway container with no network, a read-only root, no capabilities, an unprivileged user, process, memory, CPU and file limits, and a deadline; it gets no environment from the host, no mounts and no Docker socket, and the image holds neither the app nor a database driver. It reaches the tools only by messages the host validates as untrusted JSON; the bridge allows four tools, checks schemas, sizes, the line's scope and per-line call and proposal budgets; only an accepted terminal call decides, the first is final, and the verifier and policy are the agent's. Output is capped | `tests/test_sandbox.py` (no network, no secrets, no app or driver inside, read-only and noexec, fork bomb, memory and endless loop ended, output cap, forged and garbage protocol lines); `tests/test_bridge.py` (allowlist, schemas, scope, budgets, first decision final); `tests/test_probes.py` (scripted attack code per probe family through sandbox and bridge); live probe runs per arm in `results/*-probes-*.json` |
| LLM09 Misinformation | A compiled bank rule says something the owner did not mean | Code validates every compiled rule against the account, its direction, the policy envelope and the ledger's customers; rules are not applied automatically | `tests/test_rule_compiler.py`; the rule compiler table in the README |
| LLM07 System prompt leakage | The prompt reveals something exploitable | The prompt holds no secrets or credentials; the controls are in code, so knowing the prompt does not help an attacker | not applicable |
| LLM10 Unbounded consumption | A line loops or a provider fails over and over | Step, token and cost limits per line; errors that would repeat on every line stop the run. There is no run-level cost cap in code; a prepaid provider balance, an account setting, caps total spend | `tests/test_agent.py` |
| LLM03 Supply chain | A compromised dependency or CI action | Locked dependencies (`uv.lock`), GitHub Actions pinned by commit SHA, base images pinned by tag (not by digest), no agent framework or model router | CI runs `uv sync --locked` |

## What remains

- **A payload worded as an ordinary remittance passes the keyword screen.** It still meets the verifier and, outside the narrow auto-approval case, a human.
- **A payer who really is the account holder can name the wrong invoice in the reference field.** The verifier accepts the payer through the reference, and the rules pass trusts an exact reference too. Such a line can be auto-approved, because the policy checks the invoice's payer history, not the bank payee.
- **A redirect to a similarly named payer can pass the verifier's name check when the amount is exact.** For an amount that is not exact the bar is higher, after the local model matched a namesake's invoice inside the fee window. Redirects to similar names are not in the measured attack data.
- **The code-mode sandbox shares the host kernel.** Docker's runc isolation narrows it with seccomp, no capabilities and no network, but a kernel exploit would cross it; gVisor (`--runtime=runsc`) is the upgrade path for real data. Every dataset here is synthetic.
- **The model may be fooled on a line where the books allow the attacker's choice**, for example two invoices of the same payer for the same amount. That is a judgement error with a correct payer, and review is the control.
- **A local model on another machine is reached over the LAN without TLS.** It should be bound to one host by a firewall rule; nothing in this repository enforces that. It is not suitable for real data as configured.
