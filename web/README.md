# Four — AI playground

A TypeScript website using the existing Rust rules, trained value networks,
Monte Carlo rollouts, and minimax search through WebAssembly. The engine runs in
a Web Worker so searches do not block input or animation. No Python, CUDA,
inference server, or backend is needed in the browser.

## Run locally

From the repository root:

```sh
nix develop
cd web
npm ci
npm run engine
npm run dev
```

Open the URL printed by Vite. The Nix toolchain includes the
`wasm32-unknown-unknown` target. Outside Nix, install that Rust target and use
Node 22.12+ before running the npm commands.

`npm run engine` compiles the parent Rust crate and copies the available exported
models from `../models`. `small-20k.bin` is required; `large-20k.bin`,
`large-120k.bin`, and `large-1m.bin` are included when present. Re-run this command
after changing Rust or re-exporting a trained model. The browser consumes `.bin`
files, not PyTorch checkpoints. The generated assets are included so the website
can also build independently of the parent Rust project.

## Play

- Default: human Player 1 against the small-20k value network.
- Change either controller at any point between moves. Choices include humans,
  available trained models, Monte Carlo, minimax, and random play.
- Human versus AI advances the AI automatically. With two AIs, **Next move**
  advances exactly one turn.
- Choose a column or press 1–7 to play. Full columns and finished games reject
  moves. New game keeps the selected controllers and settings.
- Monte Carlo accepts 1–20,000 attempts **per legal column**, independently for
  each player. Higher budgets take longer. Progress appears while evaluating.
- Dropping discs and winning lines animate; reduced-motion preferences are
  respected.

## Column statistics

The panel retains the last AI decision while the next player considers a reply.
It reports the evaluations of the position **before that AI's move**, with the
chosen column highlighted. Full columns are marked explicitly.

Value networks report each successor state's value from the moving player's
perspective (−1 to +1), including exact terminal values. These are not calibrated
win probabilities. Monte Carlo reports wins, draws, losses, and wins minus
losses from the actual samples used to choose the move. It evaluates every legal
column for inspection; the existing tactical filtering still restricts selection
to immediate wins or safe moves when possible. Minimax reports terminal outcomes
or its existing zero evaluation for unresolved positions; random play reports
uniform probabilities. Ties are selected randomly in the browser.

## Verify and build

```sh
npm run engine
npm test
npm run build
npm run preview
```

The engine test instantiates the shipped Wasm binary with browser-style entropy,
loads all bundled models, and checks selection, Monte Carlo accounting, tactical
filtering, illegal/full columns, reset, and terminal handling. Also run
`cargo test` from the parent project when changing Rust.

Deploy `dist/` to any static HTTPS host. For a subdirectory, build with
`npm run build -- --base /connect4/`; the UI, worker, Wasm, and model URLs all
use that base. Production needs HTTPS for browser cryptographic
randomness; localhost works during development. No secrets belong in this static
bundle. The optional WebMCP tools expose the same game state and guarded actions
to supporting browsers.

## Automatic GitHub Pages deployment

The workflow in `../.github/workflows/pages.yml` builds and deploys whenever
`main` is updated on GitHub, including merged pull requests. It can also be run
manually from **Actions → Deploy website to GitHub Pages → Run workflow**.

One-time setup:

1. In the repository's **Settings → Pages → Build and deployment**, select
   **GitHub Actions** as the source.
2. Commit and push the website, the Rust changes, and the workflow to `main`.
   Include `web/package-lock.json`, `web/public/models.json`, and the exported
   `.bin` weights in `web/public/models/`. The parent `models/` directory is
   ignored by Git and is not needed by the workflow.

The workflow installs Node and Rust, rebuilds the WebAssembly engine from the
same source commit, tests it with the bundled weights, builds the TypeScript
site, and deploys `web/dist/`. It uses GitHub's built-in token, with deployment
permissions confined to the deployment job; no personal access token is needed.
The Pages configuration supplies the base path automatically, including for a
custom domain. The deployed URL appears in the workflow's `github-pages`
environment (normally `https://srtobi.github.io/connect4/` for this repository).

To reproduce the build locally using the committed model exports:

```sh
npm ci
npm run engine -- --bundled-models
npm test
npm run build -- --base /connect4/
npm run preview -- --base /connect4/
```

Open `http://localhost:4173/connect4/` for that preview. After training a new
model, run `npm run engine` locally and commit the updated public weights before
pushing to `main`.
