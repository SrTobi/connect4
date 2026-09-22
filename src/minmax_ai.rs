use std::cell::Cell;

use crate::{
    state::{Player, State},
    Ai,
};

pub struct MinMaxAi {
    depth: usize,
    player: Player,
    count: Cell<usize>,
}

impl MinMaxAi {
    pub fn column_score(&self, state: State, x: usize) -> isize {
        self.minmax(state, x, 0, isize::MIN, isize::MAX)
    }
    pub fn new(player: Player, depth: usize) -> MinMaxAi {
        MinMaxAi { depth, player, count: Cell::new(0) }
    }

    fn minmax(&self, state: State, x: usize, depth: usize, mut a: isize, mut b: isize) -> isize {
        self.count.set(self.count.get() + 1);
        let state = state.put(x);

        if let Some(win_player) = state.win_player() {
            return if win_player == self.player {
                isize::MAX - depth as isize
            } else {
                isize::MIN + depth as isize
            };
        }

        if depth == self.depth {
            return self.score(state);
        }

        if state.is_full() {
            return -1;
        }

        if state.player() == self.player {
            let mut result = isize::MIN;
            for x in state.iter_moves() {
                result = result.max(self.minmax(state, x, depth + 1, a, b));
                if result > b {
                    return result;
                }
                a = a.max(result);
            }
            result
        } else {
            let mut result = isize::MAX;
            for x in state.iter_moves() {
                result = result.min(self.minmax(state, x, depth + 1, a, b));
                if result < a {
                    return result;
                }
                b = b.min(result);
            }
            result
        }
    }

    fn score(&self, state: State) -> isize {
        0
    }
}

impl Ai for MinMaxAi {
    fn best_move(&self, state: State) -> usize {
        self.count.set(0);
        let result = state
            .iter_moves()
            .max_by_key(|&x| self.minmax(state, x, 0, isize::MIN, isize::MAX))
            .unwrap();
        println!("Count: {}", self.count.get());
        result
    }
}
