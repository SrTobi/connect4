use core::fmt;

const WIDTH: usize = 7;
const HEIGHT: usize = 6;
const FIELD_SIZE: usize = WIDTH * HEIGHT;

const FULL_MASK: u64 = !0;
const FIELD_MASK: u64 = FULL_MASK >> (64 - FIELD_SIZE);
const COLUMN_MASK: u64 = 1 << 0 | 1 << 7 | 1 << 14 | 1 << 21 | 1 << 28 | 1 << 35;

const ALL_COLUMNS_EXCEPT_LAST_MASK: u64 = COLUMN_MASK << 0
    | COLUMN_MASK << 1
    | COLUMN_MASK << 2
    | COLUMN_MASK << 3
    | COLUMN_MASK << 4
    | COLUMN_MASK << 5;
const ALL_COLUMNS_EXCEPT_LAST_TWO_MASK: u64 =
    COLUMN_MASK << 0 | COLUMN_MASK << 1 | COLUMN_MASK << 2 | COLUMN_MASK << 3 | COLUMN_MASK << 4;

const fn index(x: usize, y: usize) -> usize {
    y * WIDTH + x
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct Field {
    bits: u64,
}

impl Field {
    fn new() -> Field {
        Field { bits: 0 }
    }

    fn get(self, x: usize, y: usize) -> bool {
        (self.bits >> index(x, y)) & 1 == 1
    }

    fn set(&mut self, x: usize, y: usize, value: bool) {
        if value {
            self.bits |= 1 << index(x, y);
        } else {
            self.bits &= !(1 << index(x, y));
        }
    }

    fn stones_in_col(self, x: usize) -> usize {
        debug_assert!(x < WIDTH);
        let mask = COLUMN_MASK << x;
        let bits = self.bits & mask;
        bits.count_ones() as usize
    }

    fn is_full(self) -> bool {
        self.bits == FIELD_MASK
    }

    fn is_win(self) -> bool {
        self.is_horizontal_win()
            || self.is_vertial_win()
            || self.is_diagonal_win_nw_se()
            || self.is_diagonal_win_sw_ne()
    }

    fn is_horizontal_win(self) -> bool {
        let two_next_to_each_other = self.bits & ((self.bits & ALL_COLUMNS_EXCEPT_LAST_MASK) << 1);
        let four_next_to_each_other = two_next_to_each_other
            & ((two_next_to_each_other & ALL_COLUMNS_EXCEPT_LAST_TWO_MASK) << 2);
        // println!("{:?}", self);
        // println!();
        // println!("{:?}", Field::from(four_next_to_each_other & FIELD_MASK));
        four_next_to_each_other & FIELD_MASK != 0
    }

    fn is_vertial_win(self) -> bool {
        let two_next_to_each_other = self.bits & self.bits >> WIDTH;
        let four_next_to_each_other =
            two_next_to_each_other & (two_next_to_each_other >> 2 * WIDTH);
        four_next_to_each_other != 0
    }

    fn is_diagonal_win_nw_se(self) -> bool {
        let two_next_to_each_other =
            self.bits & ((self.bits & ALL_COLUMNS_EXCEPT_LAST_MASK) << (WIDTH + 1));
        let four_next_to_each_other = two_next_to_each_other
            & ((two_next_to_each_other & ALL_COLUMNS_EXCEPT_LAST_TWO_MASK) << (2 * WIDTH + 2));
        four_next_to_each_other & FIELD_MASK != 0
    }

    fn is_diagonal_win_sw_ne(self) -> bool {
        let two_next_to_each_other =
            self.bits & ((self.bits & ALL_COLUMNS_EXCEPT_LAST_MASK) >> (WIDTH - 1));
        let four_next_to_each_other = two_next_to_each_other
            & ((two_next_to_each_other & ALL_COLUMNS_EXCEPT_LAST_TWO_MASK) >> (2 * WIDTH - 2));
        four_next_to_each_other & FIELD_MASK != 0
    }
}

impl From<u64> for Field {
    fn from(field: u64) -> Field {
        debug_assert!(field & FIELD_MASK == field, "field is too large");
        Field { bits: field }
    }
}

impl fmt::Debug for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in (0..HEIGHT).rev() {
            for x in 0..WIDTH {
                let c = if self.get(x, y) { 'X' } else { '.' };
                write!(f, "{}", c)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum Player {
    A,
    B,
}

impl Player {
    pub fn other(self) -> Player {
        match self {
            Player::A => Player::B,
            Player::B => Player::A,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::A => write!(f, "A"),
            Player::B => write!(f, "B"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct State {
    player_a: Field,
    player_b: Field,
    player: Player,
}

impl State {
    pub fn new() -> State {
        State {
            player_a: Field::new(),
            player_b: Field::new(),
            player: Player::A,
        }
    }

    pub fn player(&self) -> Player {
        self.player
    }

    pub fn win_player(&self) -> Option<Player> {
        self.is_win().then(|| self.player.other())
    }

    pub fn get(&self, x: usize, y: usize) -> Option<bool> {
        if self.player_a.get(x, y) {
            Some(true)
        } else if self.player_b.get(x, y) {
            Some(false)
        } else {
            None
        }
    }

    fn occupancy(&self) -> Field {
        Field::from(self.player_a.bits | self.player_b.bits)
    }

    pub fn stones_in_col(&self, x: usize) -> usize {
        self.occupancy().stones_in_col(x)
    }

    pub fn can_put(&self, x: usize) -> bool {
        self.stones_in_col(x) < HEIGHT
    }

    pub fn is_full(&self) -> bool {
        self.occupancy().is_full()
    }

    pub fn is_win(&self) -> bool {
        match self.player {
            Player::A => self.player_b.is_win(),
            Player::B => self.player_a.is_win(),
        }
    }

    pub fn put(self, x: usize) -> State {
        debug_assert!(self.can_put(x), "column {x} is full");
        debug_assert!(!self.is_win(), "game is already won");

        let mut new_state = self;
        let field = match self.player {
            Player::A => &mut new_state.player_a,
            Player::B => &mut new_state.player_b,
        };
        let y = self.stones_in_col(x);
        field.set(x, y, true);
        new_state.player = new_state.player.other();
        new_state
    }

    pub fn iter_moves(self) -> impl Iterator<Item = usize> {
        let occupancy = self.occupancy();
        (0..WIDTH).filter(move |&x| occupancy.stones_in_col(x) < HEIGHT)
    }

    pub fn print(&self) {
        for y in (0..HEIGHT).rev() {
            print!("|");
            for x in 0..WIDTH {
                let c = match self.get(x, y) {
                    Some(true) => 'X',
                    Some(false) => 'O',
                    None => ' ',
                };
                print!("{}", c);
            }
            println!("|");
        }
        println!("+1234567+");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    fn make_random_field() -> Field {
        let mut field = Field::new();
        for _ in 0..thread_rng().gen_range(0..FIELD_SIZE) {
            let x = thread_rng().gen_range(0..WIDTH);
            let y = thread_rng().gen_range(0..HEIGHT);
            field.set(x, y, true);
        }
        field
    }

    #[test]
    fn random_field_test() {
        for _ in 0..10000 {
            let field = make_random_field();
            for x in 0..WIDTH {
                assert_eq!(ref_stones_in_col(field, x), field.stones_in_col(x));
                assert_eq!(ref_is_horizontal_win(field), field.is_horizontal_win());
                assert_eq!(ref_is_vertical_win(field), field.is_vertial_win());
                assert_eq!(
                    ref_is_diagonal_win_nw_se(field),
                    field.is_diagonal_win_nw_se()
                );
                assert_eq!(
                    ref_is_diagonal_win_sw_ne(field),
                    field.is_diagonal_win_sw_ne()
                );
            }
        }
    }

    fn ref_stones_in_col(fields: Field, x: usize) -> usize {
        let mut count = 0;
        for y in 0..HEIGHT {
            if fields.get(x, y) {
                count += 1;
            }
        }
        count
    }

    // checks if there are 4 consecutive pieces in a row
    fn ref_is_horizontal_win(field: Field) -> bool {
        for y in 0..HEIGHT {
            for x in 0..WIDTH - 3 {
                if field.get(x, y)
                    && field.get(x + 1, y)
                    && field.get(x + 2, y)
                    && field.get(x + 3, y)
                {
                    return true;
                }
            }
        }
        false
    }

    // checks if there are 4 consecutive pieces in a column
    fn ref_is_vertical_win(field: Field) -> bool {
        for x in 0..WIDTH {
            for y in 0..HEIGHT - 3 {
                if field.get(x, y)
                    && field.get(x, y + 1)
                    && field.get(x, y + 2)
                    && field.get(x, y + 3)
                {
                    return true;
                }
            }
        }
        false
    }

    // checks if there are 4 consecutive pieces in a diagonal
    fn ref_is_diagonal_win_nw_se(field: Field) -> bool {
        for x in 0..WIDTH - 3 {
            for y in 0..HEIGHT - 3 {
                if field.get(x, y)
                    && field.get(x + 1, y + 1)
                    && field.get(x + 2, y + 2)
                    && field.get(x + 3, y + 3)
                {
                    return true;
                }
            }
        }
        false
    }

    // checks if there are 4 consecutive pieces in a diagonal
    fn ref_is_diagonal_win_sw_ne(field: Field) -> bool {
        for x in 0..WIDTH - 3 {
            for y in 3..HEIGHT {
                if field.get(x, y)
                    && field.get(x + 1, y - 1)
                    && field.get(x + 2, y - 2)
                    && field.get(x + 3, y - 3)
                {
                    return true;
                }
            }
        }
        false
    }
}
