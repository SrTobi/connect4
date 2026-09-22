"""NumPy/ctypes bridge to the authoritative Rust rules (cargo build --release)."""
import ctypes as C
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / 'target/release/libconnect_4.so'
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')

def library():
    if not LIBRARY.exists():
        raise RuntimeError('Run cargo build --release before training.')
    lib = C.CDLL(str(LIBRARY))
    for name, args, result in [
        ('c4_new', [C.c_size_t], C.c_void_p),
        ('c4_free', [C.c_void_p], None),
        ('c4_reset', [C.c_void_p, C.c_size_t], C.c_int),
        ('c4_observe', [C.c_void_p, F32, F32, I32], C.c_int),
        ('c4_step', [C.c_void_p, I32, I32], C.c_int),
        ('c4_monte_carlo', [C.c_void_p, C.c_size_t, C.c_size_t], C.c_int),
        ('c4_values', [F32, C.c_size_t, F32, C.c_size_t, F32], C.c_int),
        ('c4_model_values', [U8, C.c_size_t, F32, C.c_size_t, F32], C.c_int),
    ]:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib

class Games:
    def __init__(self, count):
        self.lib = library()
        self.count = count
        self.handle = self.lib.c4_new(count)
        if not self.handle:
            raise ValueError('parallel games must be between 1 and 4096')

    def close(self):
        if self.handle:
            self.lib.c4_free(self.handle)
            self.handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def reset(self, index):
        if self.lib.c4_reset(self.handle, index): raise ValueError('invalid game index')

    def observe(self):
        boards = np.empty((self.count, 84), np.float32)
        successors = np.empty((self.count, 7, 84), np.float32)
        status = np.empty((self.count, 7), np.int32)
        if self.lib.c4_observe(self.handle, boards, successors, status):
            raise RuntimeError('observation failed')
        return boards, successors, status

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        if actions.shape != (self.count,): raise ValueError('one action per game required')
        results = np.empty(self.count, np.int32)
        if self.lib.c4_step(self.handle, actions, results) or (results < 0).any():
            raise ValueError('illegal action or terminal game; reset before stepping')
        return results

    def monte_carlo(self, index, attempts):
        if attempts < 1: raise ValueError('attempts must be positive')
        action = self.lib.c4_monte_carlo(self.handle, index, attempts)
        if action < 0: raise ValueError('no move available')
        return action
