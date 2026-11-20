const byId = (id) => document.getElementById(id);
let groups = new Map();
let current;
let shown = 0;

function option(value, label) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}

function setReport(report) {
  if (report.schema_version !== 1 || report.split !== "development-pilot" ||
      !report.backend?.label || !report.episodes?.length || !report.metrics) {
    throw new Error("Choose a report from the QueueLab pilot runner.");
  }
  const nextGroups = new Map();
  for (const episode of report.episodes) {
    if (typeof episode.opening?.customer_message !== "string" ||
        !Array.isArray(episode.turns) || !episode.verdict?.outcome) {
      throw new Error("The report is missing an investigation trace.");
    }
    if (!nextGroups.has(episode.opening_id)) nextGroups.set(episode.opening_id, []);
    nextGroups.get(episode.opening_id).push(episode);
  }
  groups = nextGroups;
  const tickets = [...groups].map(([id, siblings]) => option(id, siblings[0].opening.customer_message));
  byId("ticket").replaceChildren(...tickets);
  byId("ticket").disabled = false;
  byId("scenario").disabled = false;
  byId("status").textContent = `${report.backend.label} · ${report.metrics.successes}/${report.metrics.attempted} evidence-backed outcomes · saved replay, not live inference`;
  selectTicket();
  byId("case").hidden = false;
}

function selectTicket() {
  const siblings = groups.get(byId("ticket").value);
  byId("scenario").replaceChildren(...siblings.map((_, index) => option(index, `Scenario ${index + 1}`)));
  selectScenario();
}

function selectScenario() {
  current = groups.get(byId("ticket").value)[Number(byId("scenario").value)];
  shown = 0;
  byId("complaint").textContent = current.opening.customer_message;
  byId("turns").replaceChildren();
  byId("verdict").hidden = true;
  byId("advance").disabled = current.turns.length === 0;
}

function advance() {
  const turn = current.turns[shown];
  if (!turn) return;
  const item = document.createElement("li");
  const label = document.createElement("strong");
  const action = turn.action;
  label.textContent = action?.action ?
    [action.action, action.source ?? action.fact ?? action.finding].filter(Boolean).join(" · ") :
    "Invalid action";
  const description = document.createElement("p");
  description.textContent = turn.result.value ?? turn.result.error ??
    (turn.result.accepted_terminal ? `Proposed next step: ${action.next_step.replaceAll("_", " ")}` : "No record available.");
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Show action and evidence";
  const raw = document.createElement("pre");
  raw.textContent = `${turn.raw}\n\n${JSON.stringify(turn.result, null, 2)}`;
  details.append(summary, raw);
  item.append(label, description, details);
  byId("turns").append(item);
  shown += 1;
  byId("advance").disabled = shown === current.turns.length;
}

byId("ticket").addEventListener("change", selectTicket);
byId("scenario").addEventListener("change", selectScenario);
byId("advance").addEventListener("click", advance);
byId("reveal").addEventListener("click", () => {
  const { outcome, reason, expected_finding: finding, expected_next_step: step } = current.verdict;
  const verdict = byId("verdict");
  const title = document.createElement("strong");
  title.textContent = `Verdict: ${outcome}`;
  const details = document.createElement("p");
  details.textContent = `Expected: ${finding.replaceAll("_", " ")} → ${step.replaceAll("_", " ")}. ${reason}`;
  verdict.replaceChildren(title, details);
  verdict.hidden = false;
});

byId("upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    setReport(JSON.parse(await file.text()));
  } catch (error) {
    byId("status").textContent = `Couldn't open the report: ${error.message}`;
  }
});

fetch("pilot.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(setReport)
  .catch(() => {
    byId("status").textContent = "Open a saved report, or serve this directory over HTTP to load the bundled replay.";
  });
