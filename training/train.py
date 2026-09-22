"""Self-play fitted value iteration with a replay buffer and target network."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from game import Games
from model import ValueNet, action_values, choose, DEFAULT_HIDDEN, checkpoint_hidden, validate_hidden

class Replay:
    def __init__(self, capacity):
        self.capacity, self.size, self.index = capacity, 0, 0
        # Binary planes need only one byte per cell (~34 MB at 50k states).
        self.boards = np.empty((capacity, 84), np.uint8)
        self.next = np.empty((capacity, 7, 84), np.uint8)
        self.status = np.empty((capacity, 7), np.int8)

    def add(self, boards, successors, status):
        for b, n, s in zip(boards, successors, status):
            self.boards[self.index], self.next[self.index], self.status[self.index] = b, n, s
            self.index = (self.index + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)

    def sample(self, count, rng, device):
        ids = rng.integers(self.size, size=count)
        boards = self.boards[ids].astype(np.float32)
        successors = self.next[ids].astype(np.float32)
        status = self.status[ids].copy()
        # Horizontal reflection preserves gravity. Reflect both board columns
        # AND action indices; rotations and vertical flips are not valid.
        flip = rng.random(count) < 0.5
        boards[flip] = boards[flip].reshape(-1, 2, 6, 7)[..., ::-1].reshape(-1, 84)
        successors[flip] = successors[flip].reshape(-1, 7, 2, 6, 7)[:, ::-1, :, :, ::-1].reshape(-1, 7, 84)
        status[flip] = status[flip, ::-1]
        return tuple(torch.as_tensor(x, device=device) for x in (boards, successors, status))


def save(path, model, target, optimizer, games, updates, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'target': target.state_dict(),
                'optimizer': optimizer.state_dict(), 'games': games, 'updates': updates,
                'config': vars(args), 'hidden': list(model.hidden), 'format_version': 2}, tmp)
    tmp.replace(path)
    model.export(path.with_suffix('.bin'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games', type=int, default=20000, help='additional games for this run')
    p.add_argument('--hidden', type=int, nargs='+', help='hidden widths (default: 512 512 256; resume uses checkpoint architecture)')
    p.add_argument('--parallel', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--replay-size', type=int, default=50000)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--tau', type=float, default=0.01)
    p.add_argument('--epsilon-start', type=float, default=1.0)
    p.add_argument('--epsilon-end', type=float, default=0.1)
    p.add_argument('--exploration-games', type=int, default=15000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', default='models/value.pt')
    p.add_argument('--resume', help='resume weights, target and optimizer; replay starts fresh')
    p.add_argument('--save-every', type=int, default=1000)
    args = p.parse_args()
    if not (args.games > 0 and 1 <= args.parallel <= 4096 and args.batch_size > 0
            and args.replay_size >= args.batch_size and args.exploration_games > 0
            and args.save_every > 0 and args.lr > 0 and 0 < args.tau <= 1
            and 0 <= args.epsilon_end <= args.epsilon_start <= 1):
        p.error('invalid training sizes, learning rate, tau or exploration settings')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu explicitly for CPU training')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=True) if args.resume else None
    try:
        hidden = validate_hidden(args.hidden) if args.hidden is not None else (
            checkpoint_hidden(checkpoint) if checkpoint is not None else DEFAULT_HIDDEN)
        if checkpoint is not None and hidden != checkpoint_hidden(checkpoint):
            p.error('--hidden must match the resumed checkpoint; train a new model to change its size')
    except ValueError as error:
        p.error(str(error))
    model = ValueNet(hidden).to(args.device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    games = updates = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        target.load_state_dict(checkpoint['target'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        games, updates = checkpoint['games'], checkpoint['updates']
    stop = games + args.games
    next_save = games + args.save_every
    replay = Replay(args.replay_size)
    start = time.monotonic()
    last_log = start
    losses = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Training on {args.device}; hidden={hidden}; parameters={sum(p.numel() for p in model.parameters()):,}; target: {stop} completed games', flush=True)
    with Games(args.parallel) as env, output.with_suffix('.jsonl').open('a') as log:
        while games < stop:
            boards, successors, status = env.observe()
            epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * min(1, games / args.exploration_games)
            actions = choose(model, successors, status, args.device, rng, epsilon)
            replay.add(boards, successors, status)
            results = env.step(actions)
            for i in np.flatnonzero(results):
                games += 1
                env.reset(int(i))
            if replay.size >= args.batch_size:
                b, n, s = replay.sample(args.batch_size, rng, args.device)
                # Negamax Bellman target, gamma=1 for this finite zero-sum game.
                # We learn V(s), NOT a separate output for every action.
                desired = action_values(target, n, s).max(dim=1).values
                prediction = model(b)
                loss = torch.nn.functional.smooth_l1_loss(prediction, desired)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for slow, fast in zip(target.parameters(), model.parameters()):
                        slow.lerp_(fast, args.tau)
                updates += 1
                losses.append(float(loss.detach()))
            now = time.monotonic()
            if now - last_log >= 5 or games >= stop:
                row = {'games': games, 'updates': updates, 'epsilon': epsilon,
                       'loss': float(np.mean(losses)) if losses else None,
                       'seconds': round(now-start, 2)}
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row)+'\n'); log.flush()
                losses.clear(); last_log = now
            if games >= next_save or games >= stop:
                save(output, model, target, optimizer, games, updates, args)
                next_save = games + args.save_every
    print(f'Saved {output} and {output.with_suffix(".bin")}', flush=True)

if __name__ == '__main__': main()
