pub mod minmax_ai;
pub mod monte_carlo_ai;
pub mod state;
mod training;
pub mod value_ai;
mod web;

pub trait Ai {
    fn best_move(&self, state: state::State) -> usize;
}
pub mod evaluation;
