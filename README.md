# Connect Four

Rust game with Monte Carlo, minimax, and a learned **state-value** AI.

## Setup and play

```sh
nix develop
cargo build --release
cargo run --release                              # human A vs Monte Carlo B
cargo run --release -- --ai minmax --depth 4
cargo run --release -- --ai value --model models/value-large.bin
cargo run --release -- --ai value --model models/value-large.bin --self-play
```

The value AI requires an exported model (train one below). It runs entirely in
Rust; Python and CUDA are only needed for training. Run commands from the repo
root. The native training bridge currently targets Linux (`libconnect_4.so`).

## Train a state value

```sh
cargo build --release
python training/train.py --hidden 512 512 256 --games 20000 --device cuda --output models/value-large.pt
```

The default environment uses cached CUDA-enabled PyTorch packages. The trainer
requires CUDA by default; pass `--device cpu` explicitly to train without a GPU.
For this small network, CPU may also be competitive: much of self-play consists
of game simulation and small inference batches.

Default architecture: **84 → 512 ReLU → 512 ReLU → 256 ReLU → 1 tanh**
(437,761 parameters). Use `--hidden 128 128` for the original 27,521-parameter
baseline, or supply 1–8 layer widths between 1 and 4096. Larger models need more
compute and do not automatically play better; compare at the same training
budget. Changing architecture requires a new run; it cannot resume smaller
weights. An omitted `--hidden` automatically uses the saved architecture on
resume. Old checkpoints without architecture metadata use 128/128.

The 84 inputs are two binary
6×7 planes: the player to move, then their opponent. Each plane is row-major,
starting at the bottom row. Output `V(s)` estimates the outcome for the player to
move. A single shared network plays both sides.

Move choice evaluates every legal successor, using `-V(successor)` because the
opponent moves next. A winning move scores exactly +1, a draw exactly 0; illegal
moves are masked. Immediate wins are preferred even if tanh saturates on another
move. There is no hand-written blocking rule: other tactics must be learned.
Ties are randomized. Exploration occasionally chooses a uniformly random legal
move during training; evaluation has no exploration.

Training uses self-play **fitted value iteration**:

1. Run 128 games in parallel through the original Rust `State` implementation.
2. Store visited boards, their legal successors and terminal statuses in a
   50,000-position replay buffer. No duplicate Python game rules are used.
3. Sample 256 positions and train the scalar state value toward
   `max(terminal reward or -V_target(successor))` over legal moves.
4. Use Huber loss, Adam, gradient clipping, and a target network updated by
   `target ← 0.99 target + 0.01 online` after each training step.
5. Randomly mirror samples horizontally, including successor action indices.

The discount is 1 because Connect Four is finite and zero-sum. This is a
value-based method, not a policy network or DQN with seven learned outputs.
The target network and replay reduce instability, but strength is not guaranteed
by a lower training loss. Measure performance against fixed opponents.

## Checkpoints and evaluation

Each save writes:

- `models/value-large.pt`: model, target model, architecture, optimizer, counters and training settings.
- `models/value-large.bin`: versioned architecture and little-endian float32 weights for Rust inference.
- `models/value-large.jsonl`: training metrics (appended across runs).

Models and Python caches are ignored by Git. Checkpoints overwrite the selected
output; use different output names to retain older opponents. Saving is atomic
per file. A short training run is a pipeline check, not a strong trained player.

```sh
# Continue for another 20,000 games, preserving the previous checkpoint.
python training/train.py --resume models/value-large.pt --games 20000 --output models/value-large-v2.pt

# Native Rust evaluation: .bin exports, parallel CPU games, progress bar.
cargo run --release -- eval models/value-large-v2.bin --other models/value-large.bin --games 100
cargo run --release -- eval models/value-large.bin --opponent monte-carlo --attempts 100 --games 100
cargo run --release -- eval models/value-large.bin --opponent random --games 100
```

Evaluation alternates seats and reports wins/draws/losses separately for each
seat, always from the first model's perspective. Use an even number of games.
The Rust evaluator loads models once, shares their weights across workers and
runs entirely on CPU, with no Python, PyTorch, or CUDA runtime dependency. Always
use `--release`: neural inference is much slower in debug builds.

The default is up to 8 workers; adjust with `--threads N`. The progress bar shows
completed games, elapsed time and ETA on stderr. Redirecting stdout preserves
clean JSON results (`> results.json`). Non-interactive stderr receives occasional
progress lines rather than terminal control sequences. Use `--no-progress` to
suppress progress. Results include elapsed seconds and games per second.

`--seed 123` controls model tie-breaking, random opponents, and Monte Carlo
rollouts. Each game has its own seed, so changing the thread count leaves the
match results unchanged within the same build. Reproducibility across different
CPU/compiler versions is not guaranteed because floating-point results may vary.

The original Python evaluator remains available for `.pt` checkpoints:

```sh
python training/evaluate.py models/value-large-v2.pt --opponent checkpoint --other models/value-large.pt --games 100
```

Python uses different random streams and numerical kernels, so Python and Rust
need not play identical games. The Python evaluator's Monte Carlo RNG is not
controlled by its `--seed`; native Rust evaluation seeds it explicitly.

Resume restores both networks, Adam state and game/update counters; the replay
buffer, in-progress games and RNG restart. It is not an exact continuation.
`--games` counts additional games; parallel completions can overshoot by up to
`--parallel - 1`. The exploration schedule uses the total game counter. Use
`--exploration-games`, `--epsilon-start` and `--epsilon-end` to adjust it.
The default seed is 42; CUDA results are not guaranteed bit-for-bit reproducible.

## Verification

```sh
cargo test
cargo build --release
python -m unittest discover -s training -v
```

Tests cover evaluation seat accounting, thread-independent results, progress
output separation, argument validation, board perspective, terminal handling,
illegal moves, reflection, replay wraparound, and numerical agreement between exported Rust inference and
PyTorch on hundreds of legal game positions. Rebuild the release library after
changing Rust code before running Python tests or training.

For an architecture comparison with matched training settings:

```sh
python training/train.py --hidden 128 128 --games 20000 --parallel 64 --seed 42 --output models/small-20k.pt
python training/train.py --hidden 512 512 256 --games 20000 --parallel 64 --seed 42 --output models/large-20k.pt
python training/compare.py models/small-20k.pt models/large-20k.pt --games 100 --attempts 100
```

The report in `models/comparison.json` records parameter counts, completed games,
optimizer updates, training settings and seat-specific match results. Equal game
budgets need not produce equal update counts because game lengths differ. A
single training seed and 100 matches are exploratory, not a conclusive ranking.

The C ABI is internal to `training/game.py`: it requires live handles and correctly
sized contiguous buffers. `C4V2` model files contain a 32-bit little-endian dimension count,
then that many 32-bit dimensions (including 84 input and 1 output), then each
layer's float32 weights/bias in PyTorch order. Hidden activations are ReLU and
the output is tanh. Rust validates dimensions, exact weight count and finite
weights. Legacy `C4V1` files and original Python checkpoints remain supported.

## Browser playground

The TypeScript website in [`web/`](web/README.md) reuses the Rust engine through
WebAssembly. It includes animated play, configurable controllers on both sides,
manual AI-versus-AI stepping, and per-column AI statistics.

```sh
nix develop
cd web
npm ci
npm run engine
npm run dev
```
