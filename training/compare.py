"""Compare checkpoint sizes and training budgets against the same opponent settings."""
import argparse
import json
from pathlib import Path
import torch
from evaluate import evaluate
from model import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', nargs='+')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--attempts', type=int, default=100)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--output', default='models/comparison.json')
    a = p.parse_args()
    if a.games < 2 or a.games % 2 or a.attempts < 1:
        p.error('games must be positive and even; attempts must be positive')
    torch.set_num_threads(1)
    report = {'opponent': 'monte-carlo', 'attempts_per_move': a.attempts,
              'evaluation_games_per_model': a.games, 'evaluation_seed': a.seed,
              'note': 'Rust Monte Carlo randomness is not seeded; single-run results are exploratory.',
              'models': []}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for checkpoint_path in a.checkpoints:
        model = load_model(checkpoint_path, 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print(f'Evaluating {checkpoint_path} ...', flush=True)
        entry = {'checkpoint': checkpoint_path, 'hidden': list(model.hidden),
                 'parameters': sum(p.numel() for p in model.parameters()),
                 'training_games': checkpoint['games'], 'training_updates': checkpoint['updates'],
                 'training_config': checkpoint.get('config', {}),
                 'results': evaluate(model, 'monte-carlo', a.games, 'cpu', a.seed, a.attempts)}
        report['models'].append(entry)
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(entry['results']), flush=True)
    print(f'Saved {path}', flush=True)

if __name__ == '__main__': main()
