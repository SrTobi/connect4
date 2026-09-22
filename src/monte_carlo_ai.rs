use rand::{seq::IteratorRandom, thread_rng};

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

    // Prefer wins, then moves that do not let the opponent win immediately.
    // If every move loses, keep all legal moves available for simulation.
    fn tactical_moves(state: State) -> impl Iterator<Item = usize> {
        let mut winning_moves = 0u8;
        let mut safe_moves = 0u8;
        for x in state.iter_moves() {
            let next = state.put(x);
            if next.is_win() {
                winning_moves |= 1 << x;
            } else if !next.iter_moves().any(|y| next.put(y).is_win()) {
                safe_moves |= 1 << x;
            }
        }

        let preferred = if winning_moves != 0 {
            winning_moves
        } else {
            safe_moves
        };
        state
            .iter_moves()
            .filter(move |&x| preferred == 0 || preferred & (1 << x) != 0)
    }

    fn score(&self, state: State) -> isize {
        let mut wins = 0;
        let mut losses = 0;
        let mut rng = thread_rng();

        for _ in 0..self.attempts {
            let mut state = state;
            while !state.is_full() && state.win_player().is_none() {
                let x = Self::tactical_moves(state).choose(&mut rng).unwrap();
                state = state.put(x);
            }
            if let Some(win_player) = state.win_player() {
                if win_player == self.player {
                    wins += 1;
                } else {
                    losses += 1;
                }
            }
        }
        wins - losses
    }
}

impl Ai for MonteCarloAi {
    fn best_move(&self, state: State) -> usize {
        let mut moves = Self::tactical_moves(state).peekable();
        let first = moves.next().expect("no legal moves available");
        if state.put(first).is_win() || moves.peek().is_none() {
            return first;
        }
        std::iter::once(first)
            .chain(moves)
            .max_by_key(|&x| self.score(state.put(x)))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use the same one-based column numbers as the game's display.
    fn position(columns: &[usize]) -> State {
        columns
            .iter()
            .fold(State::new(), |state, x| state.put(x - 1))
    }

    #[test]
    fn takes_a_win_instead_of_blocking() {
        let state = position(&[1, 7, 2, 7, 3, 7]);
        assert_eq!(
            MonteCarloAi::tactical_moves(state).collect::<Vec<_>>(),
            vec![3]
        );
        assert_eq!(MonteCarloAi::new(Player::A, 0).best_move(state), 3);
    }

    #[test]
    fn blocks_the_review_regression_position() {
        let state = position(&[
            4, 3, 3, 6, 6, 7, 7, 6, 4, 3, 3, 6, 4, 1, 7, 6, 6, 2, 3, 1, 2, 7, 2,
        ]);
        assert_eq!(
            MonteCarloAi::tactical_moves(state).collect::<Vec<_>>(),
            vec![3]
        );
        // The block must be guaranteed, independent of sampling budget or RNG.
        assert_eq!(MonteCarloAi::new(Player::B, 0).best_move(state), 3);
    }

    #[test]
    fn retains_legal_moves_when_a_loss_is_unavoidable() {
        let state = position(&[2, 7, 3, 7, 4]);
        assert_eq!(
            MonteCarloAi::tactical_moves(state).collect::<Vec<_>>(),
            state.iter_moves().collect::<Vec<_>>()
        );
        assert!(state.can_put(MonteCarloAi::new(Player::B, 1).best_move(state)));
    }

    #[test]
    fn rollouts_take_the_opponents_immediate_win() {
        let state = position(&[1, 7, 2, 7, 3, 6]);
        // It is A's turn; scoring for B must still make A take its win.
        assert_eq!(MonteCarloAi::new(Player::B, 20).score(state), -20);
        assert_eq!(MonteCarloAi::new(Player::A, 20).score(state), 20);
    }
}
