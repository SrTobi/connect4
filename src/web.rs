//! Browser ABI. All handles and buffers are owned by the TypeScript worker.
//! Buffer pointers must come from web_alloc and have the documented sizes.
use crate::{
    minmax_ai::MinMaxAi,
    monte_carlo_ai::MonteCarloAi,
    state::{Player, State},
    value_ai::ValueAi,
};
use rand::{seq::IteratorRandom, thread_rng};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod entropy {
    #[link(wasm_import_module = "env")]
    extern "C" {
        fn browser_random(pointer: *mut u8, len: usize);
    }
    fn fill(bytes: &mut [u8]) -> Result<(), getrandom::Error> {
        unsafe {
            browser_random(bytes.as_mut_ptr(), bytes.len());
        }
        Ok(())
    }
    getrandom::register_custom_getrandom!(fill);
}

#[no_mangle]
pub extern "C" fn web_alloc(words: usize) -> *mut f32 {
    if words == 0 || words > 4_000_000 {
        return std::ptr::null_mut();
    }
    Box::into_raw(vec![0.0f32; words].into_boxed_slice()) as *mut f32
}
#[no_mangle]
pub unsafe extern "C" fn web_free(pointer: *mut f32, words: usize) {
    if !pointer.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            pointer, words,
        )));
    }
}
#[no_mangle]
pub extern "C" fn web_new() -> *mut State {
    Box::into_raw(Box::new(State::new()))
}
#[no_mangle]
pub unsafe extern "C" fn web_destroy(game: *mut State) {
    if !game.is_null() {
        drop(Box::from_raw(game));
    }
}
#[no_mangle]
pub unsafe extern "C" fn web_reset(game: *mut State) -> i32 {
    let Some(s) = game.as_mut() else {
        return -1;
    };
    *s = State::new();
    0
}
#[no_mangle]
pub unsafe extern "C" fn web_model(bytes: *const u8, size: usize) -> *mut ValueAi {
    if bytes.is_null() {
        return std::ptr::null_mut();
    }
    match ValueAi::from_bytes(std::slice::from_raw_parts(bytes, size)) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}
/// Output: 42 i32 cells (bottom first, 0 empty/1 A/2 B), turn (1/2),
/// result (0 ongoing/1 A wins/2 B wins/3 draw), number of played moves.
#[no_mangle]
pub unsafe extern "C" fn web_read(game: *const State, out: *mut i32) -> i32 {
    let Some(s) = game.as_ref() else {
        return -1;
    };
    if out.is_null() {
        return -1;
    }
    let mut ply = 0;
    for y in 0..6 {
        for x in 0..7 {
            let cell = match s.get(x, y) {
                None => 0,
                Some(true) => 1,
                Some(false) => 2,
            };
            *out.add(y * 7 + x) = cell;
            if cell != 0 {
                ply += 1;
            }
        }
    }
    *out.add(42) = if s.player() == Player::A { 1 } else { 2 };
    *out.add(43) = match s.win_player() {
        Some(Player::A) => 1,
        Some(Player::B) => 2,
        None if s.is_full() => 3,
        None => 0,
    };
    *out.add(44) = ply;
    0
}
#[no_mangle]
pub unsafe extern "C" fn web_move(game: *mut State, column: usize) -> i32 {
    let Some(s) = game.as_mut() else {
        return -1;
    };
    if column >= 7 || s.is_win() || s.is_full() || !s.can_put(column) {
        return -1;
    }
    let row = s.stones_in_col(column);
    *s = s.put(column);
    row as i32
}
/// kind: 1 value, 2 Monte Carlo, 3 minimax, 4 random.
/// Output: score, wins, draws, losses, eligible, immediate win. -1 = illegal.
#[no_mangle]
pub unsafe extern "C" fn web_column(
    game: *const State,
    model: *const ValueAi,
    kind: u32,
    budget: usize,
    column: usize,
    out: *mut f32,
) -> i32 {
    let Some(s) = game.as_ref() else {
        return -1;
    };
    if out.is_null() || column >= 7 || s.is_win() || s.is_full() || !s.can_put(column) {
        return -1;
    }
    for i in 0..6 {
        *out.add(i) = 0.0;
    }
    let next = s.put(column);
    *out.add(4) = 1.0;
    *out.add(5) = if next.is_win() { 1.0 } else { 0.0 };
    match kind {
        1 => {
            let Some(model) = model.as_ref() else {
                return -2;
            };
            *out = model.move_value(*s, column);
        }
        2 => {
            if !(1..=20000).contains(&budget) {
                return -2;
            }
            let result = MonteCarloAi::new(s.player(), budget).rollout_stats(next);
            *out = result.score() as f32;
            *out.add(1) = result.wins as f32;
            *out.add(2) = result.draws as f32;
            *out.add(3) = result.losses as f32;
            *out.add(4) = if MonteCarloAi::tactical_moves(*s).any(|x| x == column) {
                1.0
            } else {
                0.0
            };
        }
        3 => {
            if budget > 6 {
                return -2;
            }
            let score = MinMaxAi::new(s.player(), budget).column_score(*s, column);
            // Preserve distance-to-win ordering across the f32 ABI on both
            // 32-bit Wasm and native targets (large isize values would round).
            *out = if score >= isize::MAX - 100 {
                1_000_000.0 - (isize::MAX - score) as f32
            } else if score <= isize::MIN + 100 {
                -1_000_000.0 + (score - isize::MIN) as f32
            } else {
                score as f32
            };
        }
        4 => {
            *out = 1.0 / s.iter_moves().count() as f32;
        }
        _ => return -2,
    }
    0
}
/// Chooses from the same per-column scores displayed in the UI; no second
/// stochastic search. Monte Carlo retains the engine's tactical filtering.
#[no_mangle]
pub unsafe extern "C" fn web_choose(game: *const State, kind: u32, scores: *const f32) -> i32 {
    let Some(s) = game.as_ref() else {
        return -1;
    };
    if scores.is_null() || s.is_win() || s.is_full() {
        return -1;
    }
    if kind == 4 {
        return s.iter_moves().choose(&mut thread_rng()).unwrap() as i32;
    }
    if let Some(x) = s
        .iter_moves()
        .filter(|&x| s.put(x).is_win())
        .choose(&mut thread_rng())
    {
        return x as i32;
    }
    let legal: Vec<_> = s
        .iter_moves()
        .filter(|&x| kind != 2 || MonteCarloAi::tactical_moves(*s).any(|m| m == x))
        .collect();
    let best = legal
        .iter()
        .map(|&x| *scores.add(x))
        .fold(f32::NEG_INFINITY, f32::max);
    legal
        .into_iter()
        .filter(|&x| *scores.add(x) == best)
        .choose(&mut thread_rng())
        .map_or(-1, |x| x as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn web_game_and_statistics_match_rules() {
        unsafe {
            let game = web_new();
            let mut data = [0; 45];
            let mut stats = [0.0; 6];
            assert_eq!(web_move(game, 7), -1);
            for x in [0, 6, 1, 6, 2, 5] {
                assert!(web_move(game, x) >= 0);
            }
            assert_eq!(
                web_column(game, std::ptr::null(), 2, 23, 3, stats.as_mut_ptr()),
                0
            );
            assert_eq!(stats, [23.0, 23.0, 0.0, 0.0, 1.0, 1.0]);
            let scores = [0.0; 7];
            assert_eq!(web_choose(game, 2, scores.as_ptr()), 3);
            assert_eq!(web_move(game, 3), 0);
            web_read(game, data.as_mut_ptr());
            assert_eq!(data[43], 1);
            assert_eq!(data[44], 7);
            assert_eq!(web_move(game, 1), -1);
            web_reset(game);
            web_read(game, data.as_mut_ptr());
            assert_eq!(data[44], 0);
            web_destroy(game);
        }
    }
}
