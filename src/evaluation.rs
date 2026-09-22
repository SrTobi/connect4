//! Parallel CPU evaluation using exported weights, without Python or libtorch.
use crate::{
    monte_carlo_ai::MonteCarloAi,
    state::{Player, State},
    value_ai::ValueAi,
};
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};
use std::{
    io::{self, IsTerminal, Write},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

const HELP: &str = "Usage: connect-4 eval MODEL.bin [OPTIONS]
  --other MODEL.bin           Compare with a second exported model
  --opponent random|monte-carlo|checkpoint
                             Default: checkpoint with --other, otherwise random
  --games N                  Even number of games (default: 100)
  --attempts N               Monte Carlo simulations per move (default: 100)
  --threads N                Parallel games (default: available CPUs, at most 8)
  --seed N                   Seed for all move randomness (default: 123)
  --no-progress              Suppress the progress bar on stderr

Results are JSON, from MODEL.bin's perspective, split by starting seat.
Use .bin exports, not .pt training checkpoints. Build with --release.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Record {
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
}
impl Record {
    fn add(&mut self, other: Self) {
        self.wins += other.wins;
        self.draws += other.draws;
        self.losses += other.losses;
    }
    fn record(&mut self, winner: Option<Player>, side: Player) {
        match winner {
            None => self.draws += 1,
            Some(player) if player == side => self.wins += 1,
            Some(_) => self.losses += 1,
        }
    }
    fn json(&self) -> String {
        format!(
            "{{\"wins\": {}, \"draws\": {}, \"losses\": {}}}",
            self.wins, self.draws, self.losses
        )
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Results {
    pub first: Record,
    pub second: Record,
}
impl Results {
    pub fn total(&self) -> Record {
        let mut total = self.first;
        total.add(self.second);
        total
    }
}

pub enum Opponent {
    Random,
    MonteCarlo(usize),
    Checkpoint(ValueAi),
}

// Separate from model selection so terminal and seat accounting can be tested
// with exact legal move sequences.
fn play_game(mut choose: impl FnMut(State) -> usize) -> Option<Player> {
    let mut state = State::new();
    loop {
        let x = choose(state);
        assert!(x < 7 && state.can_put(x), "AI returned an illegal move");
        state = state.put(x);
        if let Some(winner) = state.win_player() {
            return Some(winner);
        }
        if state.is_full() {
            return None;
        }
    }
}

fn game_seed(seed: u64, index: usize) -> u64 {
    // SplitMix64: each game gets an independent, scheduling-independent seed.
    let mut x = seed.wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

pub fn evaluate(
    model: &ValueAi,
    opponent: &Opponent,
    games: usize,
    threads: usize,
    seed: u64,
) -> Result<Results, String> {
    evaluate_with_progress(model, opponent, games, threads, seed, |_| {})
}

fn evaluate_with_progress(
    model: &ValueAi,
    opponent: &Opponent,
    games: usize,
    threads: usize,
    seed: u64,
    mut progress: impl FnMut(usize),
) -> Result<Results, String> {
    if games < 2 || games % 2 != 0 {
        return Err("--games must be even and at least 2".into());
    }
    if threads == 0 {
        return Err("--threads must be positive".into());
    }
    if matches!(opponent, Opponent::MonteCarlo(0)) {
        return Err("--attempts must be positive".into());
    }
    let workers = threads.min(games);
    progress(0);
    thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let tx = tx.clone();
                scope.spawn(move || {
                    for game in (worker..games).step_by(workers) {
                        let side = if game % 2 == 0 { Player::A } else { Player::B };
                        let mut rng = StdRng::seed_from_u64(game_seed(seed, game));
                        let winner = play_game(|state| {
                            if state.player() == side {
                                model.best_move_with_rng(state, &mut rng)
                            } else {
                                match opponent {
                                    Opponent::Random => {
                                        state.iter_moves().choose(&mut rng).unwrap()
                                    }
                                    Opponent::Checkpoint(other) => {
                                        other.best_move_with_rng(state, &mut rng)
                                    }
                                    Opponent::MonteCarlo(attempts) => {
                                        MonteCarloAi::new(state.player(), *attempts)
                                            .best_move_with_rng(state, &mut rng)
                                    }
                                }
                            }
                        });
                        if tx.send((side, winner)).is_err() {
                            return;
                        }
                    }
                })
            })
            .collect();
        drop(tx);
        let mut result = Results::default();
        let mut completed = 0;
        let mut last_update = Instant::now();
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok((side, winner)) => {
                    let record = if side == Player::A {
                        &mut result.first
                    } else {
                        &mut result.second
                    };
                    record.record(winner, side);
                    completed += 1;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            if last_update.elapsed() >= Duration::from_millis(100) || completed == games {
                progress(completed);
                last_update = Instant::now();
            }
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| "evaluation worker failed".to_string())?;
        }
        if completed != games {
            return Err("evaluation ended before all games completed".into());
        }
        Ok(result)
    })
}

struct Options {
    model: String,
    other: Option<String>,
    opponent: String,
    games: usize,
    attempts: usize,
    threads: usize,
    seed: u64,
    progress: bool,
}
impl Options {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let model = args.next().ok_or("missing MODEL.bin")?;
        if model.starts_with('-') {
            return Err("expected MODEL.bin first".into());
        }
        let mut opts = Self {
            model,
            other: None,
            opponent: String::new(),
            games: 100,
            attempts: 100,
            threads: thread::available_parallelism().map_or(1, |n| n.get().min(8)),
            seed: 123,
            progress: true,
        };
        while let Some(arg) = args.next() {
            if arg == "--no-progress" {
                opts.progress = false;
                continue;
            }
            if ![
                "--other",
                "--opponent",
                "--games",
                "--attempts",
                "--threads",
                "--seed",
            ]
            .contains(&arg.as_str())
            {
                return Err(format!("unknown argument: {arg}"));
            }
            let value = args
                .next()
                .ok_or_else(|| format!("{arg} requires a value"))?;
            let invalid = || format!("invalid value for {arg}: {value}");
            match arg.as_str() {
                "--other" => opts.other = Some(value),
                "--opponent" => opts.opponent = value,
                "--games" => opts.games = value.parse().map_err(|_| invalid())?,
                "--attempts" => opts.attempts = value.parse().map_err(|_| invalid())?,
                "--threads" => opts.threads = value.parse().map_err(|_| invalid())?,
                "--seed" => opts.seed = value.parse().map_err(|_| invalid())?,
                _ => unreachable!(),
            }
        }
        if opts.opponent.is_empty() {
            opts.opponent = if opts.other.is_some() {
                "checkpoint"
            } else {
                "random"
            }
            .into();
        }
        if !["random", "monte-carlo", "checkpoint"].contains(&opts.opponent.as_str()) {
            return Err("unknown opponent".into());
        }
        if (opts.opponent == "checkpoint") != opts.other.is_some() {
            return Err("--other must be supplied exactly when using a checkpoint opponent".into());
        }
        if opts.games < 2 || opts.games % 2 != 0 {
            return Err("--games must be even and at least 2".into());
        }
        if opts.threads == 0 || opts.attempts == 0 {
            return Err("--threads and --attempts must be positive".into());
        }
        Ok(opts)
    }
}

pub fn run_cli(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args: Vec<_> = args.collect();
    if args.iter().any(|a| a == "--help") {
        println!("{HELP}");
        return Ok(());
    }
    let options = Options::parse(args.into_iter())?;
    let start = Instant::now();
    let load = |path: &str| {
        ValueAi::load(path)
            .map_err(|e| format!("Cannot load {path}: {e}. Use an exported .bin model."))
    };
    let model = load(&options.model)?;
    let opponent = match options.opponent.as_str() {
        "checkpoint" => Opponent::Checkpoint(load(options.other.as_deref().unwrap())?),
        "monte-carlo" => Opponent::MonteCarlo(options.attempts),
        _ => Opponent::Random,
    };
    let interactive = io::stderr().is_terminal();
    let mut last_bucket = None;
    let results = evaluate_with_progress(
        &model,
        &opponent,
        options.games,
        options.threads,
        options.seed,
        |done| {
            if !options.progress {
                return;
            }
            let bucket = done.saturating_mul(10) / options.games;
            if !interactive && last_bucket == Some(bucket) && done != options.games {
                return;
            }
            last_bucket = Some(bucket);
            let elapsed = start.elapsed().as_secs_f64();
            let eta = if done == 0 {
                "--".into()
            } else {
                format!(
                    "{:.0}s",
                    elapsed * (options.games - done) as f64 / done as f64
                )
            };
            let filled = done.saturating_mul(30) / options.games;
            let bar = format!(
                "[{}{}] {}/{} ({:.0}%)  elapsed {:.1}s  ETA {}",
                "=".repeat(filled),
                " ".repeat(30 - filled),
                done,
                options.games,
                100.0 * done as f64 / options.games as f64,
                elapsed,
                eta
            );
            if interactive {
                eprint!("\r{bar}\x1b[K");
                if done == options.games {
                    eprintln!();
                }
            } else {
                eprintln!("{bar}");
            }
            let _ = io::stderr().flush();
        },
    )?;
    let seconds = start.elapsed().as_secs_f64();
    println!("{{\n  \"first\": {},\n  \"second\": {},\n  \"total\": {},\n  \"games\": {},\n  \"threads\": {},\n  \"seed\": {},\n  \"seconds\": {:.3},\n  \"games_per_second\": {:.2}\n}}",
        results.first.json(), results.second.json(), results.total().json(), options.games,
        options.threads.min(options.games), options.seed, seconds, options.games as f64 / seconds.max(1e-9));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_model() -> ValueAi {
        let mut bytes = b"C4V2".to_vec();
        for n in [3u32, 84, 1, 1] {
            bytes.extend(n.to_le_bytes());
        }
        bytes.extend(vec![0u8; 87 * 4]);
        ValueAi::from_bytes(&bytes).unwrap()
    }
    fn scripted(moves: &[usize]) -> Option<Player> {
        let mut i = 0;
        let result = play_game(|_| {
            let x = moves[i];
            i += 1;
            x
        });
        assert_eq!(i, moves.len());
        result
    }
    #[test]
    fn terminal_results_and_seat_accounting() {
        assert!(scripted(&[0, 6, 1, 6, 2, 5, 3]) == Some(Player::A));
        assert!(scripted(&[6, 0, 6, 1, 5, 2, 5, 3]) == Some(Player::B));
        assert!(scripted(&[
            6, 5, 5, 6, 0, 2, 2, 0, 0, 1, 4, 6, 3, 3, 3, 1, 2, 6, 0, 0, 5, 0, 3, 1, 5, 2, 4, 4, 4,
            4, 1, 5, 2, 1, 1, 2, 4, 5, 6, 6, 3, 3
        ])
        .is_none());
        let mut r = Record::default();
        r.record(Some(Player::A), Player::A);
        r.record(Some(Player::B), Player::A);
        r.record(None, Player::A);
        assert_eq!(
            r,
            Record {
                wins: 1,
                draws: 1,
                losses: 1
            }
        );
        let mut r = Record::default();
        r.record(Some(Player::B), Player::B);
        r.record(Some(Player::A), Player::B);
        assert_eq!(
            r,
            Record {
                wins: 1,
                draws: 0,
                losses: 1
            }
        );
    }
    #[test]
    fn scheduling_does_not_change_results_for_any_opponent() {
        let model = tiny_model();
        for opponent in [
            Opponent::Random,
            Opponent::MonteCarlo(2),
            Opponent::Checkpoint(tiny_model()),
        ] {
            let serial = evaluate(&model, &opponent, 12, 1, 123).unwrap();
            assert_eq!(serial, evaluate(&model, &opponent, 12, 3, 123).unwrap());
            assert_eq!(serial, evaluate(&model, &opponent, 12, 32, 123).unwrap());
            assert_eq!(
                serial.total().wins + serial.total().draws + serial.total().losses,
                12
            );
            assert_eq!(
                serial.first.wins + serial.first.draws + serial.first.losses,
                6
            );
            assert_eq!(
                serial.second.wins + serial.second.draws + serial.second.losses,
                6
            );
        }
    }
    #[test]
    fn rejects_invalid_options() {
        for args in [
            vec![],
            vec!["a.bin", "--games", "3"],
            vec!["a.bin", "--threads", "0"],
            vec!["a.bin", "--other"],
            vec!["a.bin", "--games", "x"],
            vec!["a.bin", "--wat"],
            vec!["a.bin", "--opponent", "checkpoint"],
            vec!["a.bin", "--opponent", "random", "--other", "b.bin"],
        ] {
            assert!(Options::parse(args.into_iter().map(String::from)).is_err());
        }
        let opts =
            Options::parse(["a.bin", "--other", "b.bin"].into_iter().map(String::from)).unwrap();
        assert_eq!(opts.opponent, "checkpoint");
    }
}
