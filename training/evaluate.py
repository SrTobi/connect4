"""Evaluate a saved policy against random, Monte Carlo, or another checkpoint."""
import argparse
import json
import numpy as np
import torch
from game import Games
from model import choose, load_model


def evaluate(model, opponent, count, device, seed, attempts=100, other=None):
    rng = np.random.default_rng(seed)
    records = [{'wins': 0, 'draws': 0, 'losses': 0} for _ in range(2)]
    with Games(1) as env:
        for game in range(count):
            seat = game % 2
            env.reset(0)
            turn = 0
            while True:
                _, successors, status = env.observe()
                if turn == seat:
                    x = choose(model, successors, status, device, rng)[0]
                elif opponent == 'random':
                    x = rng.choice(np.flatnonzero(status[0] >= 0))
                elif opponent == 'monte-carlo':
                    x = env.monte_carlo(0, attempts)
                else:
                    x = choose(other, successors, status, device, rng)[0]
                result = env.step([x])[0]
                if result:
                    key = 'draws' if result == 2 else ('wins' if turn == seat else 'losses')
                    records[seat][key] += 1
                    break
                turn = 1-turn
    return {'first': records[0], 'second': records[1],
            'total': {key: sum(r[key] for r in records) for key in records[0]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--opponent', choices=['random','monte-carlo','checkpoint'], default='random')
    p.add_argument('--other')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--attempts', type=int, default=100)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--seed', type=int, default=123)
    a = p.parse_args()
    if a.games < 2 or a.games % 2 or a.attempts < 1: p.error('games must be positive and even (at least 2); attempts must be positive')
    if a.opponent == 'checkpoint' and not a.other: p.error('--other required for checkpoint opponent')
    torch.set_num_threads(1)
    model = load_model(a.checkpoint, a.device)
    other = load_model(a.other, a.device) if a.other else None
    print(json.dumps(evaluate(model, a.opponent, a.games, a.device, a.seed, a.attempts, other), indent=2))

if __name__ == '__main__': main()
