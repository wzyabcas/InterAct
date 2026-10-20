"""Run with unittest; set INTERACT_SMPLX_MODEL_DIR for real-model parity tests.

The model directory must contain smplx/SMPLX_MALE.npz. No IsaacGym or extra
fingertip calculation is needed to test these source reconstruction functions.
"""
import ast
import os
from pathlib import Path
import unittest

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'interact2mimic.py'
TREE = ast.parse(SCRIPT.read_text())


def functions(**models):
    names = {'to_cpu', 'decode_pca_pose', 'forward_smpl'}
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(np=np, torch=torch, **models)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), 'exec'), scope)
    return scope


class ConventionTests(unittest.TestCase):
    def test_pca_and_absolute_model_means(self):
        assignments = {n.targets[0].id: n.value for n in TREE.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        for gender in ('male', 'female', 'neutral'):
            for prefix, expected in (('smplx_model_', True), ('smplx12_model_', False)):
                kwargs = {k.arg: k.value for k in assignments[prefix + gender].keywords}
                self.assertIs(ast.literal_eval(kwargs['flat_hand_mean']), expected)

    def test_absolute_pose_cannot_be_reinterpreted_as_pca(self):
        fn = functions()
        pose = torch.zeros(2, 156)
        with self.assertRaises(ValueError):
            fn['decode_pca_pose'](pose, 'male', 'smplx', 12)
        with self.assertRaises(ValueError):
            fn['forward_smpl'](pose, np.zeros(10), np.zeros((2, 3)), 'male', 'smplx', 10, True)

    def test_original_output_width_is_591(self):
        allocations = [n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'data' for t in n.targets)]
        width = allocations[-1].args[0].elts[1]
        self.assertEqual(eval(compile(ast.Expression(width), str(SCRIPT), 'eval'), {}), 591)


@unittest.skipUnless(os.environ.get('INTERACT_SMPLX_MODEL_DIR'), 'Licensed model directory not provided')
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import smplx
        torch.set_num_threads(1)
        cls.frames = 3
        cls.create = staticmethod(lambda **kw: smplx.create(
            os.environ['INTERACT_SMPLX_MODEL_DIR'], model_type='smplx', gender='male',
            ext='npz', num_betas=10, batch_size=cls.frames, **kw))
        cls.absolute = cls.create(use_pca=False, flat_hand_mean=True)
        cls.pca12 = cls.create(use_pca=True, num_pca_comps=12, flat_hand_mean=False)
        cls.fn = functions(smplx10={'male': cls.absolute}, smplx12={'male': cls.pca12})

    def test_intercap_pca_equals_decoded_absolute(self):
        rng = np.random.RandomState(7)
        pose = torch.from_numpy(rng.normal(0, .12, (self.frames, 90)).astype(np.float32))
        beta = rng.normal(0, .1, 10).astype(np.float32)
        trans = rng.normal(0, .2, (self.frames, 3)).astype(np.float32)
        with torch.no_grad():
            direct = self.fn['forward_smpl'](pose, beta, trans, 'male', 'smplx', 10, True)
            decoded = self.fn['decode_pca_pose'](pose, 'male', 'smplx', 12)
            rebuilt = self.fn['forward_smpl'](decoded, beta, trans, 'male', 'smplx', 10)
        for a, b in zip(direct, rebuilt):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=0)
        print('InterCap max vertex error:', float((direct[0] - rebuilt[0]).abs().max()))

    def test_grab_preserves_mean_and_personalized_template(self):
        rng = np.random.RandomState(11)
        template = self.absolute.v_template.detach().clone()
        template[:, 0] *= 1.04
        template[:, 1] += .01 * torch.sin(torch.arange(len(template)).float() * .01)
        pca = self.create(use_pca=True, num_pca_comps=24, flat_hand_mean=False, v_template=template)
        absolute = self.create(use_pca=False, flat_hand_mean=True, v_template=template)
        poses = torch.from_numpy(rng.normal(0, .12, (self.frames, 114)).astype(np.float32))
        trans = torch.from_numpy(rng.normal(0, .2, (self.frames, 3)).astype(np.float32))
        beta = np.zeros(10, dtype=np.float32)
        decoded = torch.cat((poses[:, :66],
            poses[:, 66:90] @ pca.left_hand_components + pca.left_hand_mean,
            poses[:, 90:114] @ pca.right_hand_components + pca.right_hand_mean), dim=1)
        with torch.no_grad():
            direct = pca(global_orient=poses[:, :3], body_pose=poses[:, 3:66],
                         left_hand_pose=poses[:, 66:90], right_hand_pose=poses[:, 90:114], transl=trans)
            verts, joints = self.fn['forward_smpl'](decoded, beta, trans, 'male', 'smplx', 10,
                                                   smplx_model_override=absolute)
            generic, _ = self.fn['forward_smpl'](decoded, beta, trans, 'male', 'smplx', 10)
        torch.testing.assert_close(verts, direct.vertices, atol=2e-6, rtol=0)
        torch.testing.assert_close(joints, direct.joints, atol=2e-6, rtol=0)
        self.assertGreater(float((generic - verts).abs().max()), .001)
        print('GRAB max vertex error:', float((direct.vertices - verts).abs().max()))


if __name__ == '__main__':
    unittest.main()
