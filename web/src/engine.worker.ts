import { assetUrl } from "./assets";
import type {
  Request,
  Response,
  Position,
  MoveResult,
  ModelInfo,
  ColumnStat,
  Analysis,
} from "./types";
const scope = self as unknown as {
  onmessage: ((e: MessageEvent<Request>) => void) | null;
  postMessage: (r: Response) => void;
};
type Exports = Record<string, (...args: number[]) => number> & {
  memory: WebAssembly.Memory;
};
let api: Exports;
let game = 0;
let models: ModelInfo[] = [];
const loaded = new Map<string, number>();
function allocate(words: number) {
  const pointer = api.web_alloc(words);
  if (!pointer) throw Error("Not enough memory for the engine.");
  return pointer;
}
function read(): Position {
  const p = allocate(45);
  try {
    if (api.web_read(game, p) !== 0) throw Error("Cannot read the game.");
    const a = new Int32Array(api.memory.buffer, p, 45).slice();
    return {
      cells: Array.from(a.slice(0, 42)),
      turn: a[42],
      result: a[43],
      ply: a[44],
    };
  } finally {
    api.web_free(p, 45);
  }
}
async function model(id: string) {
  if (loaded.has(id)) return loaded.get(id)!;
  const info = models.find((m) => m.id === id);
  if (!info) throw Error("This model is unavailable.");
  const response = await fetch(assetUrl(info.url));
  if (!response.ok) throw Error(`Could not load ${info.label}. Try again.`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const words = Math.ceil(bytes.length / 4);
  const p = allocate(words);
  try {
    new Uint8Array(api.memory.buffer, p, bytes.length).set(bytes);
    const handle = api.web_model(p, bytes.length);
    if (!handle) throw Error("The model file is invalid.");
    loaded.set(id, handle);
    return handle;
  } finally {
    api.web_free(p, words);
  }
}
async function run(r: Request): Promise<Position | MoveResult> {
  if (r.type === "init") {
    models = r.models ?? [];
    const response = await fetch(assetUrl("engine.wasm"));
    if (!response.ok)
      throw Error("Could not load the game engine. Reload to retry.");
    const instantiated = await WebAssembly.instantiate(
      await response.arrayBuffer(),
      {
        env: {
          browser_random: (p: number, n: number) => {
            const memory = new Uint8Array(api.memory.buffer, p, n);
            for (let i = 0; i < n; i += 65536)
              crypto.getRandomValues(
                memory.subarray(i, Math.min(n, i + 65536)),
              );
          },
        },
      },
    );
    api = instantiated.instance.exports as Exports;
    game = api.web_new();
    if (!game) throw Error("Could not create a game.");
    // Make the default opponent ready before enabling play.
    await model("small-20k");
    return read();
  }
  if (!game) throw Error("The engine is still loading.");
  if (r.type === "reset") {
    api.web_reset(game);
    return read();
  }
  const before = read();
  if (before.result) throw Error("This game is over. Start a new game.");
  let column = r.column ?? -1;
  let analysis: Analysis | undefined;
  if (r.type === "ai") {
    const c = r.config;
    if (!c) throw Error("Missing player settings.");
    const kind =
      c.ai === "monte-carlo"
        ? 2
        : c.ai === "minmax"
          ? 3
          : c.ai === "random"
            ? 4
            : 1;
    const budget = kind === 2 ? c.attempts : c.depth;
    if (
      kind === 2 &&
      (!Number.isInteger(budget) || budget < 1 || budget > 20000)
    )
      throw Error("Choose 1–20,000 Monte Carlo attempts.");
    if (kind === 3 && (!Number.isInteger(budget) || budget < 0 || budget > 6))
      throw Error("Choose a search depth from 0 to 6.");
    const handle = kind === 1 ? await model(c.ai) : 0;
    const started = performance.now();
    const out = allocate(6);
    const scores = allocate(7);
    const columns: ColumnStat[] = [];
    try {
      for (let x = 0; x < 7; x++) {
        const result = api.web_column(game, handle, kind, budget, x, out);
        if (result === -2)
          throw Error("The AI could not evaluate this position.");
        const values = new Float32Array(api.memory.buffer, out, 6).slice();
        const legal = result === 0;
        new Float32Array(api.memory.buffer, scores, 7)[x] = legal
          ? values[0]
          : -Infinity;
        columns.push({
          column: x,
          legal,
          score: legal ? values[0] : 0,
          wins: values[1],
          draws: values[2],
          losses: values[3],
          eligible: legal && values[4] === 1,
          winning: legal && values[5] === 1,
        });
        scope.postMessage({ id: r.id, progress: (x + 1) / 7 });
      }
      column = api.web_choose(game, kind, scores);
      if (column < 0) throw Error("No legal move is available.");
      analysis = {
        side: before.turn,
        ai: c.ai,
        label:
          models.find((m) => m.id === c.ai)?.label ??
          { "monte-carlo": "Monte Carlo", minmax: "Minimax", random: "Random" }[
            c.ai
          ] ??
          c.ai,
        attempts: c.attempts,
        columns,
        chosen: column,
        elapsed: performance.now() - started,
        move: before.ply + 1,
      };
    } finally {
      api.web_free(out, 6);
      api.web_free(scores, 7);
    }
  }
  if (!Number.isInteger(column) || column < 0 || column > 6)
    throw Error("Choose a column from 1 to 7.");
  const row = api.web_move(game, column);
  if (row < 0) throw Error("That column is full.");
  return { position: read(), column, row, side: before.turn, analysis };
}
let queue = Promise.resolve();
scope.onmessage = (e) => {
  queue = queue.then(async () => {
    try {
      scope.postMessage({ id: e.data.id, result: await run(e.data) });
    } catch (error) {
      scope.postMessage({
        id: e.data.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
};
