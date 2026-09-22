use std::io::{self, Write};

use connect_4::{
    minmax_ai::MinMaxAi,
    monte_carlo_ai::MonteCarloAi,
    state::{Player, State},
    value_ai::ValueAi,
    Ai,
};

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

fn main() {
    if std::env::args().nth(1).as_deref() == Some("eval") {
        if let Err(error) = connect_4::evaluation::run_cli(std::env::args().skip(2)) {
            eprintln!("{error}\nUse connect-4 eval --help for usage.");
            std::process::exit(2);
        }
        return;
    }
    let mut ai_name = String::from("monte-carlo");
    let mut model = String::from("models/value.bin");
    let mut only_ai = false;
    let mut attempts = 1000;
    let mut depth = 4;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ai" => ai_name = args.next().expect("--ai needs a name"),
            "--model" => model = args.next().expect("--model needs a path"),
            "--self-play" => only_ai = true,
            "--attempts" => {
                attempts = args
                    .next()
                    .expect("--attempts needs a number")
                    .parse()
                    .expect("invalid attempts")
            }
            "--depth" => {
                depth = args
                    .next()
                    .expect("--depth needs a number")
                    .parse()
                    .expect("invalid depth")
            }
            "--help" => {
                println!("connect-4 [--ai monte-carlo|minmax|value] [--model models/value.bin] [--self-play] [--attempts 1000] [--depth 4]");
                println!("connect-4 eval MODEL.bin [--other MODEL.bin] [--games 100] [--threads N] (see eval --help)");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {arg}; use --help");
                std::process::exit(2);
            }
        }
    }
    let make_ai = |player| -> Box<dyn Ai> {
        match ai_name.as_str() {
            "monte-carlo" => Box::new(MonteCarloAi::new(player, attempts)),
            "minmax" => Box::new(MinMaxAi::new(player, depth)),
            "value" => Box::new(ValueAi::load(&model).unwrap_or_else(|e| {
                eprintln!("Cannot load {model}: {e}");
                std::process::exit(2);
            })),
            _ => {
                eprintln!("Unknown AI: {ai_name}");
                std::process::exit(2);
            }
        }
    };
    let mut state = State::new();
    let a_ai = make_ai(Player::A);
    let b_ai = make_ai(Player::B);

    loop {
        let x = if state.player() == Player::B || only_ai {
            if only_ai {
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
