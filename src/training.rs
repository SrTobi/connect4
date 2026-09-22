//! C ABI used only by training/game.py. Callers own buffers of the documented
//! sizes and must keep the handle alive; all calls on a handle are sequential.
use crate::{monte_carlo_ai::MonteCarloAi, state::State, value_ai::ValueAi, Ai};

pub struct Games {
    states: Vec<State>,
}
#[no_mangle]
pub extern "C" fn c4_new(count: usize) -> *mut Games {
    if count == 0 || count > 4096 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(Games {
        states: vec![State::new(); count],
    }))
}
#[no_mangle]
pub unsafe extern "C" fn c4_free(g: *mut Games) {
    if !g.is_null() {
        drop(Box::from_raw(g));
    }
}
#[no_mangle]
pub unsafe extern "C" fn c4_reset(g: *mut Games, index: usize) -> i32 {
    let Some(g) = g.as_mut() else {
        return -1;
    };
    let Some(s) = g.states.get_mut(index) else {
        return -1;
    };
    *s = State::new();
    0
}
/// Buffers: count*84 floats, count*7*84 floats, count*7 i32 statuses.
/// Status: -1 illegal, 0 ongoing, 1 winning move, 2 drawing move.
#[no_mangle]
pub unsafe extern "C" fn c4_observe(
    g: *const Games,
    boards: *mut f32,
    next: *mut f32,
    status: *mut i32,
) -> i32 {
    let Some(g) = g.as_ref() else {
        return -1;
    };
    if boards.is_null() || next.is_null() || status.is_null() {
        return -1;
    }
    for (i, s) in g.states.iter().enumerate() {
        std::ptr::copy_nonoverlapping(s.features().as_ptr(), boards.add(i * 84), 84);
        for x in 0..7 {
            *status.add(i * 7 + x) = -1;
            std::ptr::write_bytes(next.add((i * 7 + x) * 84), 0, 84);
        }
        if s.is_win() || s.is_full() {
            continue;
        }
        for x in s.iter_moves() {
            let child = s.put(x);
            *status.add(i * 7 + x) = if child.is_win() {
                1
            } else if child.is_full() {
                2
            } else {
                0
            };
            std::ptr::copy_nonoverlapping(
                child.features().as_ptr(),
                next.add((i * 7 + x) * 84),
                84,
            );
        }
    }
    0
}
/// Buffers: count i32 actions and results. Results: -1 invalid, 0 live, 1 win, 2 draw.
#[no_mangle]
pub unsafe extern "C" fn c4_step(g: *mut Games, actions: *const i32, results: *mut i32) -> i32 {
    let Some(g) = g.as_mut() else {
        return -1;
    };
    if actions.is_null() || results.is_null() {
        return -1;
    }
    for (i, s) in g.states.iter_mut().enumerate() {
        let x = *actions.add(i);
        *results.add(i) = -1;
        if !(0..7).contains(&x) || s.is_full() || s.is_win() || !s.can_put(x as usize) {
            continue;
        }
        *s = s.put(x as usize);
        *results.add(i) = if s.is_win() {
            1
        } else if s.is_full() {
            2
        } else {
            0
        };
    }
    0
}
#[no_mangle]
pub unsafe extern "C" fn c4_monte_carlo(g: *const Games, index: usize, attempts: usize) -> i32 {
    let Some(g) = g.as_ref() else {
        return -1;
    };
    let Some(s) = g.states.get(index) else {
        return -1;
    };
    if s.is_full() || s.is_win() {
        return -1;
    }
    MonteCarloAi::new(s.player(), attempts).best_move(*s) as i32
}
/// Pure forward pass for cross-language verification; count*84 input floats.
#[no_mangle]
pub unsafe extern "C" fn c4_values(
    weights: *const f32,
    weight_count: usize,
    boards: *const f32,
    count: usize,
    out: *mut f32,
) -> i32 {
    if weights.is_null() || boards.is_null() || out.is_null() {
        return -1;
    }
    let Ok(ai) = ValueAi::from_weights(std::slice::from_raw_parts(weights, weight_count).to_vec())
    else {
        return -1;
    };
    for i in 0..count {
        let features: &[f32; 84] = std::slice::from_raw_parts(boards.add(i * 84), 84)
            .try_into()
            .unwrap();
        *out.add(i) = ai.value(features);
    }
    0
}

/// Evaluate a complete versioned model, exercising the same parser as gameplay.
#[no_mangle]
pub unsafe extern "C" fn c4_model_values(
    model: *const u8,
    model_len: usize,
    boards: *const f32,
    count: usize,
    out: *mut f32,
) -> i32 {
    if model.is_null() || boards.is_null() || out.is_null() {
        return -1;
    }
    let Ok(ai) = ValueAi::from_bytes(std::slice::from_raw_parts(model, model_len)) else {
        return -1;
    };
    for i in 0..count {
        let features: &[f32; 84] = std::slice::from_raw_parts(boards.add(i * 84), 84)
            .try_into()
            .unwrap();
        *out.add(i) = ai.value(features);
    }
    0
}
