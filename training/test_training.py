import tempfile
import struct
from pathlib import Path
import unittest
import numpy as np
import torch
from game import Games, library
from model import ValueNet, action_values, choose, load_model, DEFAULT_HIDDEN
from train import Replay

class Tests(unittest.TestCase):
    def test_negamax_and_terminal_targets(self):
        class Constant(torch.nn.Module):
            def forward(self, x): return torch.full(x.shape[:-1], 0.75)
        status = torch.tensor([[0, 1, 2, -1, 0, 0, 0]])
        values = action_values(Constant(), torch.zeros(1, 7, 84), status)
        self.assertEqual(values[0,:3].tolist(), [-0.75, 1.0, 0.0])
        self.assertTrue(torch.isneginf(values[0,3]))
        self.assertEqual(values.max().item(), 1.0)

    def test_rules_perspective_and_illegal_moves(self):
        with Games(1) as env:
            env.step([2])
            b, _, _ = env.observe()
            self.assertEqual(b[0,44], 1)
            for _ in range(5): env.step([2])
            _, _, s = env.observe()
            self.assertEqual(s[0,2], -1)
            with self.assertRaises(ValueError): env.step([2])
            with self.assertRaises(ValueError): env.step([7])
            env.reset(0)
            self.assertEqual(env.observe()[0].sum(), 0)

    def test_winning_move_and_terminal_reset(self):
        with Games(1) as env:
            for x in [0,6,1,6,2,5]: env.step([x])
            _, n, s = env.observe()
            self.assertEqual(s[0,3], 1)
            model = ValueNet()
            for p in model.parameters(): p.data.zero_()
            self.assertEqual(choose(model,n,s,'cpu',np.random.default_rng(1))[0],3)
            self.assertEqual(env.step([3])[0],1)
            self.assertTrue((env.observe()[2] == -1).all())
            with self.assertRaises(ValueError): env.step([0])
            env.reset(0)
            self.assertTrue((env.observe()[2] == 0).all())

    def test_full_board_draw(self):
        moves = [6,5,5,6,0,2,2,0,0,1,4,6,3,3,3,1,2,6,0,0,5,0,
                 3,1,5,2,4,4,4,4,1,5,2,1,1,2,4,5,6,6,3,3]
        with Games(1) as env:
            for x in moves[:-1]: self.assertEqual(env.step([x])[0],0)
            _, n, s = env.observe()
            self.assertEqual(s[0,moves[-1]],2)
            values = action_values(ValueNet(),torch.from_numpy(n),torch.from_numpy(s))
            self.assertEqual(values[0,moves[-1]].item(),0)
            self.assertEqual(env.step([moves[-1]])[0],2)
            self.assertTrue((env.observe()[2] == -1).all())

    def test_export_matches_rust_on_legal_boards(self):
        torch.manual_seed(123)
        rng = np.random.default_rng(2)
        boards = []
        with Games(16) as env:
            for _ in range(30):
                b, _, s = env.observe(); boards.extend(b)
                actions = [rng.choice(np.flatnonzero(row >= 0)) for row in s]
                for i in np.flatnonzero(env.step(actions)): env.reset(int(i))
        boards = np.ascontiguousarray(boards, dtype=np.float32)
        for hidden in [(128,128), (23,), (64,32,16), DEFAULT_HIDDEN]:
            with self.subTest(hidden=hidden), tempfile.TemporaryDirectory() as folder:
                model = ValueNet(hidden).eval()
                path = Path(folder)/'model.bin'; model.export(path)
                blob = path.read_bytes(); self.assertEqual(blob[:4], b'C4V2')
                out = np.empty(len(boards),np.float32)
                data = np.frombuffer(blob,dtype=np.uint8).copy()
                self.assertEqual(library().c4_model_values(data, len(data), boards, len(boards), out),0)
                with torch.no_grad(): expected = model(torch.from_numpy(boards)).numpy()
                np.testing.assert_allclose(out,expected,atol=2e-6,rtol=2e-5)
                if hidden == (128,128):
                    # Old .bin files and checkpoints continue to load.
                    count = struct.unpack_from('<I',blob,4)[0]
                    legacy = np.frombuffer(b'C4V1'+blob[8+count*4:],dtype=np.uint8).copy()
                    self.assertEqual(library().c4_model_values(legacy,len(legacy),boards,len(boards),out),0)
                    np.testing.assert_allclose(out,expected,atol=2e-6,rtol=2e-5)
                checkpoint = Path(folder)/'model.pt'
                saved = {'model':model.state_dict()}
                if hidden != (128,128): saved['hidden'] = list(hidden)
                torch.save(saved, checkpoint)
                restored = load_model(checkpoint,'cpu')
                self.assertEqual(restored.hidden,hidden)
                with torch.no_grad():
                    np.testing.assert_allclose(restored(torch.from_numpy(boards)).numpy(),expected)

    def test_bad_architectures_and_model_files(self):
        for hidden in [(), (0,), (-1,), (4097,), (2,)*9]:
            with self.assertRaises(ValueError): ValueNet(hidden)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.bin'; ValueNet((8,)).export(path)
            good = path.read_bytes()
            # Truncated magic/header/weights, excessive dimensions, trailing bytes.
            for blob in [b'', b'C4V2', good[:10], good[:-4], good+b'x',
                         b'C4V2'+struct.pack('<I',1000000),
                         b'C4V2'+struct.pack('<IIII',3,84,0,1)]:
                data = np.frombuffer(blob,dtype=np.uint8).copy()
                self.assertEqual(library().c4_model_values(data,len(data),np.zeros((1,84),np.float32),1,np.zeros(1,np.float32)),-1)

    def test_replay_reflection_and_wraparound(self):
        with Games(1) as env:
            env.step([0]); b,n,s = env.observe()
            replay = Replay(2)
            for _ in range(3): replay.add(b,n,s)
            self.assertEqual(replay.size,2)
            rb,rn,rs = replay.sample(32,np.random.default_rng(3),'cpu')
            mirrored = b.reshape(1,2,6,7)[...,::-1].reshape(84)
            mirrored_next = n.reshape(1,7,2,6,7)[:,::-1,:,:,::-1].reshape(7,84)
            seen = set()
            for board,next_board,status in zip(rb.numpy(),rn.numpy(),rs.numpy()):
                flip = bool(board[48]); seen.add(flip)
                np.testing.assert_array_equal(board,mirrored if flip else b[0])
                np.testing.assert_array_equal(next_board,mirrored_next if flip else n[0])
                np.testing.assert_array_equal(status,s[0,::-1] if flip else s[0])
            self.assertEqual(seen,{True,False})

if __name__ == '__main__': unittest.main()
