import "./style.css";
import { assetUrl } from "./assets";
import type {
  ModelInfo,
  PlayerConfig,
  Position,
  Response,
  MoveResult,
  Analysis,
  Request,
} from "./types";
const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
 <header class="header"><a class="brand" href="${assetUrl("")}" aria-label="Four home"><span class="brand-mark"><i></i><i></i><i></i><i></i></span>Four<span class="brand-dot">.</span></a><span class="header-caption">THE AI PLAYGROUND</span><button id="reset" class="quiet-button">↻ <span>New game</span></button></header>
 <main class="layout">
 <aside class="players"><div class="section-label">THE MATCHUP</div><h1>Your move.<br><span>Or theirs.</span></h1><div id="players"></div><div class="match-note"><span class="note-icon">◎</span><p id="mode-note">Connect four in any direction. Pick a column to drop a disc.</p></div></aside>
 <section class="game-section" aria-label="Connect Four game"><div class="game-top"><div id="turn" role="status" aria-live="polite">Loading the engine…</div><span id="move-number" class="mono">MOVE 00</span></div><div class="board-wrap"><div id="column-numbers" class="column-numbers">${Array.from({ length: 7 }, (_, i) => `<span>${i + 1}</span>`).join("")}</div><div id="board" class="board" aria-label="Game board"></div><div id="column-buttons" class="column-buttons">${Array.from({ length: 7 }, (_, i) => `<button data-column="${i}" aria-label="Drop in column ${i + 1}" disabled></button>`).join("")}</div></div><div class="game-bottom"><span id="game-hint">Preparing your opponent</span><button id="next" class="primary-button" hidden>Next move <span>→</span></button></div><div id="error" role="alert" hidden></div><div class="board-legend"><span><i class="a-dot"></i>Player 1</span><span><i class="b-dot"></i>Player 2</span><span class="legend-right">7 columns. 6 rows. 4 to win.</span></div></section>
 <aside class="insight"><div class="section-label">BEHIND THE MOVE</div><div class="insight-heading"><h2>What the AI sees</h2><span id="analysis-badge" class="chip">Awaiting move</span></div><div id="analysis-summary" class="analysis-summary">After an AI plays, explore how it rated each column.</div><div id="stats" class="stats"></div><div id="analysis-note" class="analysis-note"></div></aside>
 </main><footer><span>Built to play. Open to explore.</span><span>Rust engine · Learned values &amp; Monte Carlo</span></footer>`;
const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;
let models: ModelInfo[] = [];
const players: PlayerConfig[] = [
  { ai: "human", attempts: 1000, depth: 4 },
  { ai: "small-20k", attempts: 1000, depth: 4 },
];
let position: Position = {
  cells: Array(42).fill(0),
  turn: 1,
  result: 0,
  ply: 0,
};
let busy = true;
let animating = false;
let generation = 0;
let last: Analysis | undefined;
let progress = 0;
let error = "";
let ready = false;
const worker = new Worker(new URL("./engine.worker.ts", import.meta.url), {
  type: "module",
});
let sequence = 0;
const pending = new Map<
  number,
  { resolve: (data: any) => void; reject: (error: Error) => void }
>();
worker.onmessage = (event: MessageEvent<Response>) => {
  const r = event.data;
  if (r.progress !== undefined) {
    progress = r.progress;
    renderStatus();
    return;
  }
  const p = pending.get(r.id);
  if (!p) return;
  pending.delete(r.id);
  r.error ? p.reject(Error(r.error)) : p.resolve(r.result);
};
worker.onerror = () => {
  for (const p of pending.values())
    p.reject(Error("The engine stopped. Reload the page to retry."));
  pending.clear();
  busy = false;
  error = "The engine stopped. Reload the page to retry.";
  renderStatus();
};
function request<T>(
  type: Request["type"],
  extra: Partial<Request> = {},
): Promise<T> {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...extra });
  });
}
function label(ai: string) {
  return (
    models.find((m) => m.id === ai)?.label ??
    {
      human: "Human",
      "monte-carlo": "Monte Carlo",
      minmax: "Minimax",
      random: "Random",
    }[ai] ??
    ai
  );
}
function human(side: number) {
  return players[side - 1].ai === "human";
}
function bothAI() {
  return !human(1) && !human(2);
}
function renderPlayers() {
  $("players").innerHTML = players
    .map(
      (p, i) =>
        `<section class="player-card side-${i + 1} ${position.turn === i + 1 && !position.result ? "active" : ""}" data-player="${i}"><div class="player-card-head"><span class="player-disc"></span><span>PLAYER ${i + 1}</span><span class="seat">${i === 0 ? "FIRST" : "SECOND"}</span></div><label for="ai-${i}">Controller</label><select id="ai-${i}" aria-label="Player ${i + 1} AI"><option value="human">Human</option>${models.map((m) => `<option value="${m.id}">${m.label}</option>`).join("")}<option value="monte-carlo">Monte Carlo</option><option value="minmax">Minimax</option><option value="random">Random</option></select><div class="controller-description">${p.ai === "human" ? "You call the shots." : p.ai === "monte-carlo" ? "Simulates games before choosing." : p.ai === "minmax" ? "Looks ahead for wins and losses." : p.ai === "random" ? "Every legal column has a chance." : "A value network trained through self-play."}</div>${p.ai === "monte-carlo" ? `<label for="attempts-${i}">Attempts per column</label><input id="attempts-${i}" type="number" min="1" max="20000" step="1" value="${p.attempts}" inputmode="numeric"><span class="input-note">1–20,000 · More attempts take longer</span>` : ""}${p.ai === "minmax" ? `<label for="depth-${i}">Search depth</label><select id="depth-${i}">${[1, 2, 3, 4, 5, 6].map((d) => `<option value="${d}" ${p.depth === d ? "selected" : ""}>${d + 1} moves ahead</option>`).join("")}</select>` : ""}</section>`,
    )
    .join("");
  players.forEach((p, i) => {
    const select = $<HTMLSelectElement>(`ai-${i}`);
    select.value = p.ai;
    select.disabled = busy || animating || !ready;
    select.onchange = () => {
      players[i].ai = select.value;
      error = "";
      renderPlayers();
      renderStatus();
      scheduleAI();
    };
    const attempts = document.getElementById(
      `attempts-${i}`,
    ) as HTMLInputElement | null;
    if (attempts) {
      attempts.disabled = busy || animating;
      attempts.oninput = () => {
        const n = Number(attempts.value);
        if (Number.isInteger(n) && n >= 1 && n <= 20000)
          players[i].attempts = n;
      };
      attempts.onchange = () => {
        const n = Number(attempts.value);
        if (!Number.isInteger(n) || n < 1 || n > 20000) {
          attempts.setCustomValidity("Enter an integer from 1 to 20,000.");
          attempts.reportValidity();
          attempts.value = String(p.attempts);
          attempts.setCustomValidity("");
          return;
        }
        players[i].attempts = n;
      };
    }
    const depth = document.getElementById(
      `depth-${i}`,
    ) as HTMLSelectElement | null;
    if (depth) {
      depth.disabled = busy || animating;
      depth.onchange = () => (players[i].depth = Number(depth.value));
    }
  });
}
function winningCells(): Set<number> {
  const found = new Set<number>();
  if (!position.result || position.result === 3) return found;
  for (let y = 0; y < 6; y++)
    for (let x = 0; x < 7; x++)
      for (const [dx, dy] of [
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1],
      ]) {
        const ids = Array.from({ length: 4 }, (_, n) => [
          x + dx * n,
          y + dy * n,
        ]);
        if (
          ids.every(
            ([a, b]) =>
              a >= 0 &&
              a < 7 &&
              b >= 0 &&
              b < 6 &&
              position.cells[b * 7 + a] === position.result,
          )
        )
          ids.forEach(([a, b]) => found.add(b * 7 + a));
      }
  return found;
}
function renderBoard(drop?: { column: number; row: number }) {
  const wins = winningCells();
  $("board").innerHTML = Array.from({ length: 42 }, (_, n) => {
    const x = n % 7,
      y = 5 - Math.floor(n / 7),
      side = position.cells[y * 7 + x];
    const falling = drop?.column === x && drop.row === y;
    return `<div class="slot">${side ? `<div class="disc side-${side} ${falling ? "falling" : ""} ${wins.has(y * 7 + x) ? "winning" : ""}" style="--fall:${-(6 - y) * 105}%" aria-label="Player ${side} disc at column ${x + 1}, row ${y + 1}">${falling ? "<span></span>" : ""}</div>` : ""}</div>`;
  }).join("");
  $("column-buttons").className = `column-buttons side-${position.turn}`;
  document.querySelectorAll<HTMLButtonElement>("[data-column]").forEach((b) => {
    const x = Number(b.dataset.column);
    b.disabled =
      !ready ||
      busy ||
      animating ||
      !!position.result ||
      !human(position.turn) ||
      position.cells[35 + x] !== 0;
    b.onclick = () => humanMove(x);
  });
}
function renderStatus() {
  const ended = !!position.result;
  const ai = !human(position.turn);
  let title = "";
  if (!ready) title = "Loading the engine…";
  else if (ended)
    title =
      position.result === 3
        ? "A perfectly matched draw."
        : `Player ${position.result} connects four!`;
  else if (animating) title = `Player ${position.turn === 1 ? 2 : 1} played`;
  else if (busy) title = `${label(players[position.turn - 1].ai)} is thinking…`;
  else
    title = human(position.turn)
      ? `Player ${position.turn} · your turn`
      : `Player ${position.turn} · ready to play`;
  $("turn").innerHTML =
    `<span class="turn-disc side-${ended && position.result !== 3 ? position.result : position.turn} ${busy ? "thinking" : ""}"></span><span>${title}</span>`;
  $("move-number").textContent =
    `MOVE ${String(position.ply).padStart(2, "0")}`;
  $("game-hint").textContent = ended
    ? "New game, same curiosity."
    : animating
      ? ""
      : busy && ready
        ? `Evaluating columns · ${Math.round(progress * 100)}%`
        : bothAI()
          ? "One click. One decision."
          : ai
            ? "Your opponent plays automatically."
            : "Choose a column above or use keys 1–7.";
  const next = $<HTMLButtonElement>("next");
  next.hidden = ended || !ready || (!bothAI() && !error);
  next.disabled = busy || animating || human(position.turn);
  next.innerHTML = `${error ? "Retry AI move" : "Next move"} <span>→</span>`;
  $("mode-note").textContent = bothAI()
    ? "AI vs AI. Press Next move to advance one turn and inspect its reasoning."
    : "Connect four in any direction. Pick a column to drop a disc.";
  $<HTMLButtonElement>("reset").disabled = !ready;
  $("error").hidden = !error;
  $("error").textContent = error;
  document.querySelectorAll<HTMLButtonElement>("[data-column]").forEach((b) => {
    b.disabled =
      !ready ||
      busy ||
      animating ||
      ended ||
      !human(position.turn) ||
      position.cells[35 + Number(b.dataset.column)] !== 0;
  });
}
function renderStats() {
  if (!last) {
    $("stats").innerHTML = Array.from(
      { length: 7 },
      (_, x) =>
        `<div class="stat-row empty"><span class="column-label">${x + 1}</span><div class="empty-line"></div><span class="stat-value">—</span></div>`,
    ).join("");
    $("analysis-badge").textContent = "Awaiting move";
    $("analysis-summary").textContent =
      "After an AI plays, explore how it rated each column.";
    $("analysis-note").textContent =
      "Scores stay visible while you plan your reply.";
    return;
  }
  const a = last;
  const mc = a.ai === "monte-carlo",
    minmax = a.ai === "minmax",
    random = a.ai === "random";
  $("analysis-badge").textContent = `MOVE ${String(a.move).padStart(2, "0")}`;
  $("analysis-summary").innerHTML =
    `<div class="analysis-player"><i class="side-${a.side}"></i><strong>${a.label}</strong><span class="mono">${a.elapsed < 1000 ? Math.round(a.elapsed) + " ms" : (a.elapsed / 1000).toFixed(1) + " s"}</span></div><p>Played column <strong>${a.chosen + 1}</strong> · ${mc ? "wins − losses" : minmax ? "search score" : random ? "selection probability" : "value for the player who moved"}</p>`;
  $("stats").innerHTML = a.columns
    .map((s) => {
      const forced = minmax && Math.abs(s.score) > 900000;
      const value = !s.legal
        ? "Full"
        : forced
          ? s.score > 0
            ? "Win"
            : "Loss"
          : mc
            ? `${s.score > 0 ? "+" : ""}${s.score}`
            : random
              ? `${Math.round(s.score * 100)}%`
              : `${s.score >= 0 ? "+" : ""}${s.score.toFixed(2)}`;
      const normalized = mc
        ? s.score / a.attempts
        : minmax
          ? forced
            ? Math.sign(s.score)
            : 0
          : s.score;
      const width = random ? s.score * 100 : Math.abs(normalized) * 50;
      return `<div class="stat-row ${a.chosen === s.column ? "chosen" : ""} ${!s.legal ? "illegal" : ""} ${!s.eligible && s.legal ? "unsafe" : ""}"><span class="column-label">${s.column + 1}</span><div class="stat-main"><div class="stat-meter ${random ? "probability" : ""}">${s.legal ? `<i class="${normalized < 0 ? "negative" : "positive"}" style="width:${Math.min(100, width)}%;left:${random ? 0 : normalized < 0 ? 50 - width : 50}%"></i>` : ""}</div>${mc && s.legal ? `<div class="counts"><span class="win-count">${s.wins} W</span><span>${s.draws} D</span><span class="loss-count">${s.losses} L</span></div>` : ""}${s.legal && !s.eligible ? '<span class="excluded">Excluded by tactical check</span>' : ""}</div><span class="stat-value">${value}${a.chosen === s.column ? "<small>PLAYED</small>" : ""}</span></div>`;
    })
    .join("");
  $("analysis-note").innerHTML = mc
    ? `${a.attempts.toLocaleString()} simulations per legal column. Higher net wins are better. Tactical checks prefer immediate wins and safe moves.`
    : minmax
      ? "Terminal wins and losses take priority. Unresolved positions currently score 0."
      : random
        ? "All legal columns have the same chance. No search or learned evaluation."
        : "−1 favors the opponent. +1 favors the player who moved. These are learned estimates, not win probabilities.";
}
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
async function move(type: "human" | "ai", column?: number) {
  if (!ready || busy || animating || position.result) return;
  const token = generation;
  busy = true;
  progress = 0;
  error = "";
  renderPlayers();
  renderStatus();
  try {
    const result = await request<MoveResult>(
      type,
      type === "ai"
        ? { config: { ...players[position.turn - 1] } }
        : { column },
    );
    if (token !== generation) return;
    position = result.position;
    if (result.analysis) last = result.analysis;
    busy = false;
    animating = true;
    renderBoard({ column: result.column, row: result.row });
    renderStats();
    renderStatus();
    await delay(
      matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 560,
    );
    if (token !== generation) return;
    animating = false;
    renderPlayers();
    renderBoard();
    renderStatus();
    scheduleAI();
  } catch (e) {
    if (token !== generation) return;
    busy = false;
    animating = false;
    error = e instanceof Error ? e.message : String(e);
    renderPlayers();
    renderStatus();
  }
}
function humanMove(column: number) {
  if (human(position.turn)) return move("human", column);
  return Promise.reject(Error("It is an AI turn."));
}
function scheduleAI() {
  const token = generation;
  if (
    !ready ||
    busy ||
    animating ||
    position.result ||
    bothAI() ||
    human(position.turn) ||
    error
  )
    return;
  setTimeout(() => {
    if (token === generation && !bothAI() && !human(position.turn) && !error)
      void move("ai");
  }, 300);
}
async function reset() {
  if (!ready) return;
  const token = ++generation;
  busy = true;
  animating = false;
  error = "";
  renderPlayers();
  renderStatus();
  try {
    position = await request<Position>("reset");
    if (token !== generation) return;
    last = undefined;
    busy = false;
    renderBoard();
    renderPlayers();
    renderStats();
    renderStatus();
    scheduleAI();
  } catch (e) {
    busy = false;
    error = String(e);
    renderStatus();
  }
}
$("reset").onclick = () => void reset();
$("next").onclick = () => void move("ai");
document.addEventListener("keydown", (e) => {
  if (
    e.target instanceof HTMLInputElement ||
    e.target instanceof HTMLSelectElement ||
    e.ctrlKey ||
    e.metaKey ||
    e.altKey
  )
    return;
  if (
    /^[1-7]$/.test(e.key) &&
    human(position.turn) &&
    !busy &&
    !animating &&
    !position.result
  ) {
    e.preventDefault();
    void humanMove(Number(e.key) - 1);
  }
});
renderPlayers();
renderBoard();
renderStats();
renderStatus();
async function init() {
  try {
    const response = await fetch(assetUrl("models.json"));
    if (!response.ok) throw Error("Could not load the available models.");
    models = await response.json();
    position = await request<Position>("init", { models });
    ready = true;
    busy = false;
    renderPlayers();
    renderBoard();
    renderStatus();
  } catch (e) {
    busy = false;
    error = e instanceof Error ? e.message : String(e);
    renderStatus();
  }
}
void init();
// Expose the same actions to browsers implementing WebMCP; no separate game state.
const context = (
  document as Document & {
    modelContext?: { registerTool: (tool: unknown) => void };
  }
).modelContext;
if (context) {
  context.registerTool({
    name: "read_connect_four",
    description:
      "Read the current board, player controllers, and last AI column statistics.",
    inputSchema: { type: "object", properties: {} },
    annotations: { readOnlyHint: true },
    execute: () => ({ position, players, last, busy: busy || animating }),
  });
  context.registerTool({
    name: "play_connect_four_column",
    description: "Drop a human disc into a legal column numbered 1 through 7.",
    inputSchema: {
      type: "object",
      properties: { column: { type: "integer", minimum: 1, maximum: 7 } },
      required: ["column"],
      additionalProperties: false,
    },
    execute: async (input: { column: number }) => {
      if (
        !Number.isInteger(input.column) ||
        input.column < 1 ||
        input.column > 7 ||
        busy ||
        animating ||
        !ready ||
        position.result ||
        !human(position.turn) ||
        position.cells[35 + input.column - 1]
      )
        throw Error("That move is not available.");
      await humanMove(input.column - 1);
      return position;
    },
  });
  context.registerTool({
    name: "advance_connect_four_ai",
    description: "Play the next AI turn in a match with two AI controllers.",
    inputSchema: { type: "object", properties: {} },
    execute: async () => {
      if (!bothAI() || busy || animating || !ready || position.result)
        throw Error("Next AI move is not available.");
      await move("ai");
      return { position, last };
    },
  });
}
