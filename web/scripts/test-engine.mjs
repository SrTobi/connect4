// Exercise the actual browser binary and its memory/entropy ABI without a UI.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';

const binary = readFileSync(new URL('../public/engine.wasm', import.meta.url));
let api;
const { instance } = await WebAssembly.instantiate(binary, {
  env: { browser_random: (p, n) => webcrypto.getRandomValues(new Uint8Array(api.memory.buffer, p, n)) },
});
api = instance.exports;
const game = api.web_new();
const data = api.web_alloc(45);
const out = api.web_alloc(6);
const scores = api.web_alloc(7);
function read() {
  assert.equal(api.web_read(game, data), 0);
  return [...new Int32Array(api.memory.buffer, data, 45)];
}
function column(x, kind = 2, budget = 37, model = 0) {
  const result = api.web_column(game, model, kind, budget, x, out);
  return { result, values: [...new Float32Array(api.memory.buffer, out, 6)] };
}
try {
  assert.equal(read()[44], 0);
  assert.equal(api.web_move(game, 7), -1);
  for (let x = 0; x < 7; x++) {
    const { result, values: [score, wins, draws, losses] } = column(x);
    assert.equal(result, 0);
    assert.equal(wins + draws + losses, 37);
    assert.equal(score, wins - losses);
  }
  assert.equal(column(0, 2, 0).result, -2);
  assert.equal(column(0, 2, 20001).result, -2);

  // Every shipped network loads through the same pointer ABI as the worker.
  const models = JSON.parse(readFileSync(new URL('../public/models.json', import.meta.url)));
  for (const info of models) {
    const bytes = readFileSync(new URL('../public' + info.url, import.meta.url));
    const words = Math.ceil(bytes.length / 4);
    const p = api.web_alloc(words);
    new Uint8Array(api.memory.buffer, p, bytes.length).set(bytes);
    const model = api.web_model(p, bytes.length);
    api.web_free(p, words);
    assert.ok(model, info.id);
    const values = Array.from({ length: 7 }, (_, x) => {
      const c = column(x, 1, 0, model);
      assert.equal(c.result, 0);
      assert.ok(Number.isFinite(c.values[0]) && Math.abs(c.values[0]) <= 1);
      return c.values[0];
    });
    new Float32Array(api.memory.buffer, scores, 7).set(values);
    const chosen = api.web_choose(game, 1, scores);
    assert.equal(values[chosen], Math.max(...values));
  }

  // A full column is rejected, and reset clears the board.
  for (let i = 0; i < 6; i++) assert.equal(api.web_move(game, 0), i);
  assert.equal(api.web_move(game, 0), -1);
  assert.equal(column(0).result, -1);
  api.web_reset(game);
  assert.equal(read()[44], 0);

  // Tactical filtering must override even an artificially attractive score.
  for (const x of [0, 6, 1, 6, 4, 6]) api.web_move(game, x);
  new Float32Array(api.memory.buffer, scores, 7).fill(100);
  new Float32Array(api.memory.buffer, scores, 7)[6] = -100;
  assert.equal(api.web_choose(game, 2, scores), 6);
  api.web_reset(game);
  for (const x of [0, 6, 1, 6, 2, 5]) api.web_move(game, x);
  assert.deepEqual(column(3).values, [37, 37, 0, 0, 1, 1]);
  assert.equal(column(3, 3, 4).values[0], 1000000);
  assert.equal(api.web_choose(game, 2, scores), 3);
  assert.equal(api.web_move(game, 3), 0);
  assert.equal(read()[43], 1);
  assert.equal(read()[44], 7);
  assert.equal(api.web_move(game, 4), -1);
  assert.equal(api.web_choose(game, 2, scores), -1);
  console.log(`Browser engine checks passed (${models.length} models, Monte Carlo counts, tactical moves, full columns, reset, game over).`);
} finally {
  api.web_free(data, 45);
  api.web_free(out, 6);
  api.web_free(scores, 7);
  api.web_destroy(game);
}
