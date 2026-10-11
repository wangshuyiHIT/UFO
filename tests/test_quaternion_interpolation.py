"""CPU regressions for source-frame and intermediate motion orientations."""
import unittest

import torch

from humanoidverse.utils.torch_utils import (
    axis_angle_to_quaternion, matrix_to_quaternion, quaternion_to_matrix, slerp,
)


class QuaternionInterpolationTest(unittest.TestCase):
    def test_nearby_rotation_keeps_source_endpoints(self):
        # Representative slow root motion: the old small-angle branch returned
        # a midpoint even at t=0, producing a 0.19 mm reset FK inconsistency.
        q0 = torch.tensor([[.999826967716217, .0028521569911390543,
                            .0183755811303854, .0003931785759050399]])
        q1 = torch.tensor([[.9998233318328857, .0028303933795541525,
                            .018571002408862114, .0005509481998160481]])
        for time, expected in ((0., q0), (1., q1)):
            actual = slerp(q0, q1, torch.tensor([[time]]))
            expected = expected / expected.norm(dim=-1, keepdim=True)
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
            torch.testing.assert_close(actual.norm(dim=-1), torch.ones(1))

    def test_small_rotation_respects_time_and_quaternion_sign(self):
        angles = torch.tensor([[0., 0., 0.], [0., 0., .001]], dtype=torch.float64)
        q = axis_angle_to_quaternion(angles)
        times = torch.tensor([[0.], [.2], [.7], [1.]], dtype=torch.float64)
        actual = slerp(q[:1].expand(4, -1), -q[1:].expand(4, -1), times)
        expected = axis_angle_to_quaternion(torch.cat([torch.zeros(4, 2), .001*times], dim=-1))
        torch.testing.assert_close(actual, expected, atol=3e-12, rtol=0)

    def test_large_rotation_and_roundoff_are_finite_and_unit_length(self):
        q = axis_angle_to_quaternion(torch.tensor([[0., 0., 0.], [0., 0., 2.]]))
        times = torch.tensor([[0.], [.25], [.5], [1.]])
        actual = slerp(q[:1].expand(4, -1), q[1:].expand(4, -1), times)
        expected = axis_angle_to_quaternion(torch.cat([torch.zeros(4, 2), 2*times], dim=-1))
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
        roundtrip = matrix_to_quaternion(quaternion_to_matrix(q)) * (1. + 2e-7)
        same = slerp(roundtrip, -roundtrip, torch.full((2, 1), .3))
        self.assertTrue(torch.isfinite(same).all())
        torch.testing.assert_close(same.norm(dim=-1), torch.ones(2), atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
