use std::io::{self, Write};

use state::State;

use crate::state::Player;

mod minmax_ai;
mod monte_carlo_ai;
mod state;

trait Ai {
    fn best_move(&self, state: State) -> usize;
}

fn prompt(state: State) -> usize {
    loop {
        print!("Enter a number: ");
        io::stdout().flush().ok();

        let mut input_line = String::new();
        io::stdin()
            .read_line(&mut input_line)
            .expect("Failed to read line");
        match input_line.trim().parse::<usize>() {
            Ok(x) if x < 1 || x > 7 || !state.can_put(x - 1) => {
                println!("Cannot put in column {x}")
            }
            Ok(x) => return x - 1,
            Err(_) => println!("Please enter a number"),
        }
    }
}

const ONLY_AI: bool = false;

fn main() {
    let mut state = State::new();
    let a_ai = monte_carlo_ai::MonteCarloAi::new(Player::A, 20_000);
    let b_ai = monte_carlo_ai::MonteCarloAi::new(Player::B, 1000);

    loop {
        let x = if state.player() == Player::B || ONLY_AI {
            if ONLY_AI {
                state.print();
                println!();
            }
            match state.player() {
                Player::A => a_ai.best_move(state),
                Player::B => b_ai.best_move(state),
            }
        } else {
            state.print();
            println!();
            prompt(state)
        };

        state = state.put(x);
        println!();
        if let Some(win_player) = state.win_player() {
            state.print();
            println!("Player {} wins!", win_player);
            return;
        } else if state.is_full() {
            state.print();
            println!("Draw!");
            return;
        }
    }
}
