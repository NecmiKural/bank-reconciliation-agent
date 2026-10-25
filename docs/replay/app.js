// The replay viewer for docs/replay/<slice>.json, written by `recon export`.
// Recorded model text and planted payloads are hostile by design, so every string reaches
// the page as a text node: no string is ever parsed as markup. tests/test_gate.py fails if
// this file uses an HTML-parsing property.

const REPO = "https://github.com/NecmiKural/bank-reconciliation-agent";
const SLICES = {
  "A-slice": "deepseek-flash, one conversation per line",
  "A-slice-sandbox": "deepseek-flash, code mode in our sandbox",
  "A-slice-claude": "claude-haiku-5-5, one conversation per line",
  "A-slice-ptc": "claude-haiku-5-5, programmatic tool calling",
};
const OUTCOMES = {
  correct: "Correct",
  sent_to_person: "Sent to a person",
  wrong_match: "Wrong match",
};
const GOALS = {
  redirect: "redirect the payment to",
  over_allocate: "allocate the payment to extra invoices",
  preapproval: "have the match approved without review",
  pii_leak: "get personal data copied into the rationale",
};
const STATEMENT_INTRO = {
  sandbox:
    "One conversation decided every open line. The model's only tool was run_python: each cell ran in our Docker sandbox, and every tool call the code made passed the bridge's checks.",
  ptc: "One conversation decided every open line with Claude's programmatic tool calling: the code ran in Anthropic's code execution container, and every tool call it made came back here to the bridge's checks.",
};

const $ = (id) => document.getElementById(id);
const state = { slice: "A-slice", view: "lines", line: null, step: 1, outcome: "all" };
const docs = new Map();
let build = null;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value === true) node.setAttribute(key, "");
    else if (value !== false && value != null) node.setAttribute(key, String(value));
  }
  node.append(...nodes(children));
  return node;
}

// Children as given by the step builders: nested lists, with null, false or "" for "nothing".
const nodes = (list) => list.flat(Infinity).filter((child) => child != null && child !== false && child !== "");

const words = (name) => String(name).replaceAll("_", " ");
const pre = (text, cls = "") => el("pre", { class: cls, tabindex: "0" }, String(text));
const json = (value) => pre(JSON.stringify(value, null, 1));

async function load(slice) {
  if (!docs.has(slice)) {
    const response = await fetch(`${slice}.json`);
    if (!response.ok) throw new Error(`${slice}.json: HTTP ${response.status}`);
    docs.set(slice, await response.json());
  }
  return docs.get(slice);
}

// Deep links: #slice, #slice/statement or #slice/line/step, each part checked against known ids.
function route() {
  let parts = [];
  try {
    parts = decodeURIComponent(location.hash.slice(1)).split("/");
  } catch {
    parts = [];
  }
  return { slice: Object.hasOwn(SLICES, parts[0]) ? parts[0] : "A-slice", rest: parts.slice(1) };
}

function writeHash() {
  const parts = [state.slice];
  if (state.view === "statement") parts.push("statement");
  else if (state.line) parts.push(state.line, String(state.step));
  history.replaceState(null, "", `#${parts.join("/")}`);
}

async function show(slice, rest = []) {
  let doc;
  try {
    doc = await load(slice);
  } catch (error) {
    $("alert").textContent = `Could not load the recording: ${error.message}`;
    $("alert").hidden = false;
    return;
  } finally {
    document.querySelector("main").removeAttribute("aria-busy");
  }
  $("alert").hidden = true;
  state.slice = slice;
  $("slice").value = slice;
  const ids = doc.lines.map((line) => line.id);
  if (rest[0] === "statement" && doc.statement) {
    state.view = "statement";
  } else {
    state.view = "lines";
    state.line = ids.includes(rest[0]) ? rest[0] : ids.includes(state.line) ? state.line : ids[0];
    const step = /^\d{1,3}$/.test(rest[1] || "") ? Number(rest[1]) : 1;
    state.step = Math.min(Math.max(step, 1), steps(doc, current(doc)).length);
  }
  renderSlice(doc);
  writeHash();
}

const current = (doc) => doc.lines.find((line) => line.id === state.line);

function renderSlice(doc) {
  const date = doc.recorded_at.slice(0, 10);
  $("slice-info").textContent =
    `Recorded on ${date} with ${doc.model} at commit ${doc.recorded_commit}: the first 40 lines ` +
    `the rules pass leaves open in synthetic business ${doc.business}, replayed through the ` +
    `whole pipeline.`;
  renderHeadline(doc);
  renderBuild(doc);
  $("views").hidden = !doc.statement;
  $("show-lines").setAttribute("aria-pressed", String(state.view === "lines"));
  $("show-statement").setAttribute("aria-pressed", String(state.view === "statement"));
  $("lines-view").hidden = state.view !== "lines";
  $("statement-view").hidden = state.view !== "statement";
  if (state.view === "statement") renderStatement(doc);
  else {
    renderRows(doc);
    renderDetail(doc, false);
  }
}

// The README headline is the page's only aggregate claim; **bold** is kept as <strong>.
function renderHeadline(doc) {
  const parts = doc.readme_headline.split("**");
  $("headline").replaceChildren(
    ...parts.map((part, i) => (i % 2 ? el("strong", {}, part) : document.createTextNode(part))),
  );
  $("headline-source").textContent =
    `Verbatim from the README results at ${doc.results_at}, measured on full live runs. ` +
    `The four slices below are the 40-line recordings that CI replays on every change.`;
}

function renderBuild(doc) {
  const commit = build && /^[0-9a-f]{7,40}$/.test(build.commit) ? build.commit : null;
  $("build").replaceChildren(
    "Personal project, MIT licence. ",
    commit ? `Built from commit ${commit}; ` : "Built from a local checkout; ",
    `results at ${doc.results_at}. `,
    el("a", { href: commit ? `${REPO}/tree/${commit}` : REPO }, "Source and results on GitHub"),
  );
}

function renderRows(doc) {
  const shown = doc.lines.filter((l) => state.outcome === "all" || l.outcome === state.outcome);
  $("rows").replaceChildren(
    ...shown.map((line) => {
      const selected = line.id === state.line;
      return el(
        "tr",
        { class: selected ? "selected" : "" },
        el(
          "th",
          { scope: "row" },
          el(
            "button",
            { type: "button", "aria-current": selected ? "true" : false, onclick: () => pick(line.id) },
            line.id,
          ),
        ),
        el("td", { class: "num" }, line.amount),
        el("td", {}, line.decision.decision),
        el("td", {}, el("span", { class: "outcome", "data-outcome": line.outcome }, OUTCOMES[line.outcome])),
        el("td", {}, line.attack ? words(line.attack.goal) : ""),
        el("td", {}, line.policy.auto ? "auto" : "review"),
      );
    }),
  );
  $("empty").hidden = shown.length > 0;
}

function pick(id) {
  state.line = id;
  state.step = 1;
  const doc = docs.get(state.slice);
  renderRows(doc);
  renderDetail(doc, true);
  writeHash();
}

function steps(doc, line) {
  const list = [{ title: "the bank line", body: () => bankLine(doc, line) }];
  if (line.turns) {
    line.turns.forEach((turn, i) => list.push({ title: `model turn ${i + 1}`, body: () => modelTurn(turn) }));
  } else {
    line.calls.forEach((call, i) =>
      list.push({ title: `tool call ${i + 1}, ${call.tool}`, body: () => codeCall(call) }),
    );
  }
  list.push({ title: "the decision and the policy", body: () => decisionStep(line) });
  list.push({ title: "the label and the outcome", body: () => labelStep(line) });
  return list;
}

function renderDetail(doc, focus) {
  const line = current(doc);
  if (!line) return;
  const list = steps(doc, line);
  const step = list[state.step - 1];
  $("detail-title").textContent = `Line ${line.id}`;
  $("stepper").hidden = false;
  $("step-status").textContent = `Step ${state.step} of ${list.length}: ${step.title}`;
  $("ticks").replaceChildren(
    ...list.map((_, i) => el("li", { class: i < state.step ? "done" : "" })),
  );
  $("prev").disabled = state.step === 1;
  $("next").disabled = state.step === list.length;
  const heading = el("h3", { tabindex: "-1" }, step.title[0].toUpperCase() + step.title.slice(1));
  $("step").replaceChildren(heading, ...nodes(step.body()));
  if (focus) heading.focus();
}

function move(delta) {
  const doc = docs.get(state.slice);
  const total = steps(doc, current(doc)).length;
  const next = Math.min(Math.max(state.step + delta, 1), total);
  if (next === state.step) return;
  state.step = next;
  renderDetail(doc, true);
  writeHash();
}

function bankLine(doc, line) {
  return [
    el("p", {}, "The message the model received. Payer-written fields sit inside the untrusted block, with personal data patterns redacted."),
    pre(line.message, "message"),
    el(
      "p",
      {},
      doc.statement
        ? "The rules pass left this line open. In code mode its message was part of one statement message holding every open line."
        : "The rules pass left this line open, so it went to the agent in a conversation of its own.",
    ),
  ];
}

function modelTurn(turn) {
  return [
    turn.reasoning && el("details", {}, el("summary", {}, "Model reasoning"), pre(turn.reasoning, "prose")),
    turn.text && el("p", { class: "model-text" }, turn.text),
    turn.calls.length ? null : el("p", {}, "No tool call in this turn."),
    turn.calls.map((call) => callBlock(call.tool, call.args, call.result, null)),
  ];
}

function codeCall(call) {
  const where = call.from_code ? "Made by the model's code" : "Called directly by the model";
  return [
    el(
      "p",
      {},
      `${where} during turn ${call.turn} of the statement conversation. `,
      el("button", { type: "button", class: "link", onclick: () => showTurn(call.turn) }, `Show turn ${call.turn}`),
    ),
    callBlock(call.tool, call.args, call.result, call.refusal),
  ];
}

function callBlock(tool, args, result, refusal) {
  return el("section", { class: "call" }, el("h4", {}, "Call ", el("code", {}, tool)), argsList(args), outcomeOf(tool, result, refusal));
}

function argsList(args) {
  if (!args || typeof args !== "object" || Array.isArray(args)) return json(args);
  return el(
    "dl",
    { class: "args" },
    Object.entries(args).map(([key, value]) => [
      el("dt", {}, words(key)),
      el("dd", {}, value === "" ? "(empty)" : Array.isArray(value) ? value.join(", ") : typeof value === "string" ? value : JSON.stringify(value)),
    ]),
  );
}

function outcomeOf(tool, result, refusal) {
  if (refusal) return el("p", { class: "refused" }, `The bridge refused this call (${words(refusal)}): ${result?.error ?? ""}`);
  if (result == null) return el("p", {}, "No result was recorded for this call.");
  if (typeof result !== "object") return pre(result);
  if (result.error) {
    const who = tool === "propose_match" ? "The verifier refused this proposal" : "The tool refused this call";
    return el("p", { class: "refused" }, `${who}: ${result.error}`);
  }
  if (result.status === "proposed") return el("p", { class: "accepted" }, "The verifier accepted this proposal. It goes to the policy and a reviewer, never straight to the ledger.");
  if (result.status === "escalated") return el("p", {}, "The line goes to a person.");
  if (Array.isArray(result.candidates)) return candidates(result);
  if (Array.isArray(result.exact_combinations)) {
    const combos = result.exact_combinations;
    return el("p", {}, combos.length ? `Exact combinations: ${combos.map((c) => c.join(" + ")).join("; ")}.` : "No combination adds up exactly.");
  }
  return json(result);
}

function candidates(result) {
  if (!result.candidates.length) return el("p", {}, `No open invoice found for this line of NZ$${result.line_amount}.`);
  const columns = ["invoice", "payee", "amount", "difference", "relation", "payee_similarity"];
  const figure = (c) => (["amount", "difference", "payee_similarity"].includes(c) ? "num" : "");
  return [
    el("p", { class: "quiet" }, `Candidates for a line of NZ$${result.line_amount}; code computed every difference.`),
    el(
      "div",
      { class: "table-wrap", role: "region", "aria-label": "Candidates the search returned", tabindex: "0" },
      el(
        "table",
        { class: "candidates" },
        el("thead", {}, el("tr", {}, columns.map((c) => el("th", { scope: "col", class: figure(c) }, words(c))))),
        el(
          "tbody",
          {},
          result.candidates.map((row) => el("tr", {}, columns.map((c) => el("td", { class: figure(c) }, String(row[c] ?? ""))))),
        ),
      ),
    ),
  ];
}

function decisionStep(line) {
  const d = line.decision;
  const what = d.decision === "match" ? `Match to ${d.invoice_ids.join(", ")}` : "Send to a person";
  return [
    el("p", { class: "verdict" }, what),
    d.rationale && el("figure", {}, el("figcaption", {}, "Rationale for the reviewer"), el("blockquote", {}, d.rationale)),
    el(
      "p",
      {},
      line.policy.auto
        ? `Shadow policy: would let it skip review (${line.policy.reason}).`
        : `Shadow policy: a person reviews it (${line.policy.reason}).`,
    ),
    el("p", { class: "quiet" }, "In shadow mode the policy only records what it would do. Nothing is posted without a reviewer's approval."),
  ];
}

function labelStep(line) {
  const label = line.label;
  const expected =
    label.expected === "match" ? `a match to ${label.invoice_ids.join(", ")}` : label.expected === "escalate" ? "a person" : "a bank rule";
  const attack = line.attack;
  return [
    el("p", {}, `The label expected ${expected}. Case: ${words(label.case)}.`),
    el("p", {}, "Outcome: ", el("span", { class: "outcome", "data-outcome": line.outcome }, OUTCOMES[line.outcome])),
    attack
      ? el(
          "p",
          {},
          `A planted payload on this line tries to ${GOALS[attack.goal] ?? words(attack.goal)}` +
            (attack.targets.length ? ` ${attack.targets.join(", ")}` : "") +
            `. It ${attack.succeeded ? "succeeded" : "did not succeed"}.`,
        )
      : el("p", {}, "No planted payload on this line."),
  ];
}

function renderStatement(doc) {
  $("statement-intro").textContent = STATEMENT_INTRO[doc.arm] ?? "";
  $("turns").replaceChildren(...doc.statement.map((turn, i) => turnItem(turn, i + 1)));
}

function turnItem(turn, n) {
  const heading = el("h3", { id: `turn-${n}`, tabindex: "-1" }, `Turn ${n}`);
  const quiet = !turn.text && !turn.reasoning && !turn.code.length;
  return el(
    "li",
    { class: quiet ? "turn quiet" : "turn" },
    heading,
    turn.reasoning && el("details", {}, el("summary", {}, "Model reasoning"), pre(turn.reasoning, "prose")),
    turn.text && el("p", { class: "model-text" }, turn.text),
    turn.code.map((cell) => [el("p", { class: "label" }, "Code"), pre(cell.code, "code")]),
    callList(turn.calls),
    turn.code.map((cell) => cellOutput(cell.output)),
  );
}

function cellOutput(output) {
  if (output == null) return el("p", { class: "quiet" }, "No output was recorded before the conversation ended.");
  return [
    el("p", { class: "label" }, "Output"),
    pre(output.stdout || "(nothing printed)", "output"),
    output.error && [el("p", { class: "label" }, "Error"), pre(output.error, "output error")],
    output.truncated && el("p", { class: "quiet" }, "The output was cut to the sandbox's limit."),
    output.sandbox && el("p", { class: "quiet" }, output.sandbox),
  ];
}

function callList(calls) {
  if (!calls.length) return el("p", { class: "quiet" }, "No tool call from code in this turn.");
  const text = (call) => `${call.tool} for ${call.line_id ?? "an unknown line"}`;
  if (calls.length <= 3) return el("p", { class: "calls" }, `Tool calls from code: ${calls.map(text).join("; ")}.`);
  return el(
    "details",
    {},
    el("summary", {}, `${calls.length} tool calls from code`),
    el("ul", { class: "call-list" }, calls.map((call) => el("li", {}, text(call)))),
  );
}

function showTurn(n) {
  state.view = "statement";
  const doc = docs.get(state.slice);
  renderSlice(doc);
  writeHash();
  $(`turn-${n}`)?.focus();
}

function wire() {
  $("slice").replaceChildren(
    ...Object.entries(SLICES).map(([id, label]) => el("option", { value: id }, label)),
  );
  $("slice").addEventListener("change", (event) => {
    state.line = null;
    show(event.target.value);
  });
  $("outcome").addEventListener("change", (event) => {
    state.outcome = event.target.value;
    renderRows(docs.get(state.slice));
  });
  $("show-lines").addEventListener("click", () => show(state.slice, [state.line, String(state.step)]));
  $("show-statement").addEventListener("click", () => show(state.slice, ["statement"]));
  $("prev").addEventListener("click", () => move(-1));
  $("next").addEventListener("click", () => move(1));
  // The arrow keys step only while focus is inside the stepper, and never inside a scrollable
  // block, where they scroll.
  $("stepper").addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest("pre, .table-wrap")) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      move(event.key === "ArrowLeft" ? -1 : 1);
    }
  });
  window.addEventListener("hashchange", () => {
    const { slice, rest } = route();
    show(slice, rest);
  });
}

async function start() {
  wire();
  try {
    const response = await fetch("build.json");
    build = response.ok ? await response.json() : null;
  } catch {
    build = null;
  }
  const { slice, rest } = route();
  await show(slice, rest);
}

start();
