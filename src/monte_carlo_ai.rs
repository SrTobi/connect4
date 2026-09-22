use rand::{rngs::ThreadRng, seq::IteratorRandom, thread_rng};

use crate::{
    state::{Player, State},
    Ai,
};

pub struct MonteCarloAi {
    attempts: usize,
    player: Player,
}

impl MonteCarloAi {
    pub fn new(player: Player, attempts: usize) -> MonteCarloAi {
        MonteCarloAi { attempts, player }
    }

    fn score(&self, state: State) -> isize {
        let mut wins = 0;
        let mut loose = 0;

        for _ in 0..self.attempts {
            let mut state = state;
            while !state.is_full() && state.win_player().is_none() {
                let x = state.iter_moves().choose(&mut thread_rng()).unwrap();
                state = state.put(x);
            }
            if let Some(win_player) = state.win_player() {
                if win_player == self.player {
                    wins += 2;
                } else {
                    loose += 1;
                }
            }
        }
        wins - loose
    }
}

impl Ai for MonteCarloAi {
    fn best_move(&self, state: State) -> usize {
        state
            .iter_moves()
            .max_by_key(|&x| self.score(state.put(x)))
            .unwrap()
    }
}
