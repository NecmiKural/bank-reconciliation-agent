---
title: Bank Reconciliation Agent replay
emoji: 🧾
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: true
short_description: Recorded agent runs on synthetic data, replayed
license: mit
tags:
  - agents
  - tool-calling
  - evaluation
  - synthetic-data
  - mcp
---

# Bank Reconciliation Agent replay

A static viewer for the four 40-line slices that the [bank-reconciliation-agent](https://github.com/NecmiKural/bank-reconciliation-agent) CI replays on every change: `deepseek-flash` and `claude-haiku-5-5`, each with one conversation per bank line and in code mode. Pick a line and step through what the model was sent, every tool call and its result, the verifier's answer, the decision, the shadow policy's verdict and the label.

- **Synthetic data only.** Every line, payer and invoice was generated from a fixed seed; none is a real business or person.
- **Replayed from recorded runs, not live.** Nothing on this page calls a model, and it needs no key. The files are written by `recon export`, and `tests/test_gate.py` fails if they differ from a fresh replay.
- The page loads only its own files: no third-party script, font or request.

Personal project, MIT licence.
