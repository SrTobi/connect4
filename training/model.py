"""State value, always from the player-to-move perspective."""
from pathlib import Path
import struct
import numpy as np
import torch
from torch import nn

DEFAULT_HIDDEN = (512, 512, 256)
LEGACY_HIDDEN = (128, 128)


def validate_hidden(hidden):
    hidden = tuple(hidden)
    if not 1 <= len(hidden) <= 8 or any(type(n) is not int or not 1 <= n <= 4096 for n in hidden):
        raise ValueError('use 1 to 8 hidden layers, each with 1 to 4096 neurons')
    return hidden


def checkpoint_hidden(checkpoint):
    # Original checkpoints predate architecture metadata.
    return validate_hidden(checkpoint.get('hidden', LEGACY_HIDDEN))


class ValueNet(nn.Sequential):
    def __init__(self, hidden=DEFAULT_HIDDEN):
        hidden = validate_hidden(hidden)
        dimensions = (84, *hidden, 1)
        layers = []
        for i, (inputs, outputs) in enumerate(zip(dimensions, dimensions[1:])):
            layers.extend([nn.Linear(inputs, outputs),
                           nn.Tanh() if i == len(dimensions)-2 else nn.ReLU()])
        super().__init__(*layers)
        self.hidden = hidden

    def forward(self, x):
        return super().forward(x).squeeze(-1)

    def export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        weights = np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters()])
        dimensions = (84, *self.hidden, 1)
        header = b'C4V2' + struct.pack('<I', len(dimensions))
        header += struct.pack('<' + 'I'*len(dimensions), *dimensions)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_bytes(header + weights.astype('<f4').tobytes())
        tmp.replace(path)

@torch.no_grad()
def action_values(model, successors, status):
    # Successors use the opponent's perspective. Terminal overrides must take
    # precedence over the network, especially for a win on the 42nd move.
    values = -model(successors)
    values = torch.where(status == 1, 1.0, values)
    values = torch.where(status == 2, 0.0, values)
    return values.masked_fill(status < 0, -torch.inf)

@torch.no_grad()
def choose(model, successors, status, device, rng, epsilon=0.0):
    values = action_values(model, torch.as_tensor(successors, device=device),
                           torch.as_tensor(status, device=device)).cpu().numpy()
    actions = []
    for scores, move_status in zip(values, status):
        legal = move_status >= 0
        candidates = np.flatnonzero(legal)
        if rng.random() >= epsilon:
            wins = np.flatnonzero(move_status == 1)
            candidates = wins if len(wins) else np.flatnonzero(legal & (scores == scores.max()))
        actions.append(rng.choice(candidates))
    return np.asarray(actions, dtype=np.int32)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ValueNet(checkpoint_hidden(checkpoint)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model
