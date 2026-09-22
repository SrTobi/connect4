use crate::{state::State, Ai};
use rand::{seq::IteratorRandom, thread_rng, Rng};
use std::{fs, io, path::Path};

/// Legacy C4V1 files have fixed 128/128 hidden layers. C4V2 files include
/// layer dimensions before the little-endian f32 weights (PyTorch order).
pub const WEIGHT_COUNT: usize = 84 * 128 + 128 + 128 * 128 + 128 + 128 + 1;
pub struct ValueAi {
    dimensions: Vec<usize>,
    weights: Vec<f32>,
}

fn invalid_model() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid C4V1/C4V2 value network",
    )
}

// Independent accumulators let the compiler vectorize the dot product without
// requiring CPU-specific instructions or unsafe intrinsics. A single running
// float sum creates a dependency chain and is substantially slower here.
fn dot(input: &[f32], weights: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut inputs = input.chunks_exact(8);
    let mut rows = weights.chunks_exact(8);
    for (xs, ws) in inputs.by_ref().zip(rows.by_ref()) {
        for i in 0..8 {
            lanes[i] += xs[i] * ws[i];
        }
    }
    let mut total: f32 = lanes.iter().sum();
    for (x, w) in inputs.remainder().iter().zip(rows.remainder()) {
        total += x * w;
    }
    total
}

impl ValueAi {
    pub fn from_weights(weights: Vec<f32>) -> io::Result<Self> {
        Self::with_dimensions(vec![84, 128, 128, 1], weights)
    }

    fn with_dimensions(dimensions: Vec<usize>, weights: Vec<f32>) -> io::Result<Self> {
        if !(3..=10).contains(&dimensions.len())
            || dimensions.first() != Some(&84)
            || dimensions.last() != Some(&1)
            || dimensions.iter().any(|&n| n == 0 || n > 4096)
        {
            return Err(invalid_model());
        }
        let expected: usize = dimensions.windows(2).map(|d| d[0] * d[1] + d[1]).sum();
        if weights.len() != expected || weights.iter().any(|v| !v.is_finite()) {
            return Err(invalid_model());
        }
        Ok(Self {
            dimensions,
            weights,
        })
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::from_bytes(&fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let (dimensions, offset) = match bytes.get(..4) {
            Some(b"C4V1") => (vec![84, 128, 128, 1], 4),
            Some(b"C4V2") => {
                let count = u32::from_le_bytes(
                    bytes
                        .get(4..8)
                        .ok_or_else(invalid_model)?
                        .try_into()
                        .unwrap(),
                ) as usize;
                if !(3..=10).contains(&count) {
                    return Err(invalid_model());
                }
                let end = 8 + count * 4;
                let dimensions = bytes
                    .get(8..end)
                    .ok_or_else(invalid_model)?
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                    .collect();
                (dimensions, end)
            }
            _ => return Err(invalid_model()),
        };
        if (bytes.len() - offset) % 4 != 0 {
            return Err(invalid_model());
        }
        let weights = bytes[offset..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Self::with_dimensions(dimensions, weights)
    }

    pub fn value(&self, features: &[f32; 84]) -> f32 {
        let mut values = features.to_vec();
        let mut offset = 0;
        for (layer, dims) in self.dimensions.windows(2).enumerate() {
            let (inputs, outputs) = (dims[0], dims[1]);
            let mut next = vec![0.0; outputs];
            let final_layer = layer == self.dimensions.len() - 2;
            for (row, value) in next.iter_mut().enumerate() {
                *value = self.weights[offset + inputs * outputs + row];
                let weights = &self.weights[offset + row * inputs..offset + (row + 1) * inputs];
                *value += dot(&values, weights);
                *value = if final_layer {
                    value.tanh()
                } else {
                    value.max(0.0)
                };
            }
            values = next;
            offset += inputs * outputs + outputs;
        }
        values[0]
    }

    pub fn move_value(&self, state: State, x: usize) -> f32 {
        let next = state.put(x);
        if next.is_win() {
            1.0
        } else if next.is_full() {
            0.0
        } else {
            -self.value(&next.features())
        }
    }
}

impl ValueAi {
    pub fn best_move_with_rng(&self, state: State, rng: &mut impl Rng) -> usize {
        assert!(!state.is_win() && !state.is_full(), "game is over");
        // tanh can saturate at +/-1: a predicted win must not tie away an
        // actual terminal win.
        if let Some(x) = state
            .iter_moves()
            .filter(|&x| state.put(x).is_win())
            .choose(rng)
        {
            return x;
        }
        let moves: Vec<_> = state
            .iter_moves()
            .map(|x| (x, self.move_value(state, x)))
            .collect();
        let best = moves
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        moves
            .iter()
            .filter(|(_, v)| *v == best)
            .choose(rng)
            .unwrap()
            .0
    }
}

impl Ai for ValueAi {
    fn best_move(&self, state: State) -> usize {
        self.best_move_with_rng(state, &mut thread_rng())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_win_and_opponent_perspective() {
        let mut w = vec![0.0; WEIGHT_COUNT];
        w[WEIGHT_COUNT - 1] = 0.5;
        let ai = ValueAi::from_weights(w).unwrap();
        assert!((ai.move_value(State::new(), 0) + 0.5f32.tanh()).abs() < 1e-6);
        let s = [0, 6, 1, 6, 2, 5]
            .iter()
            .fold(State::new(), |s, &x| s.put(x));
        assert_eq!(ai.best_move(s), 3);
        assert_eq!(ai.move_value(s, 3), 1.0);
    }
    #[test]
    fn rejects_invalid_models() {
        assert!(ValueAi::from_weights(vec![]).is_err());
        assert!(ValueAi::from_weights(vec![f32::NAN; WEIGHT_COUNT]).is_err());
    }
    #[test]
    fn encoding_switches_perspective() {
        let s = State::new().put(2);
        assert_eq!(s.features()[42 + 2], 1.0);
        assert_eq!(s.features().iter().sum::<f32>(), 1.0);
        let s = s.put(3);
        assert_eq!(s.features()[2], 1.0);
        assert_eq!(s.features()[42 + 3], 1.0);
    }
}
