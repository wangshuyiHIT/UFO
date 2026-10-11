"""Independent MuJoCo checks of off-grid reference pose and velocity semantics."""
import tempfile
import unittest
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_lib.morphology_kinematics import MorphologyKinematics, interpolate_root
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.torch_utils import slerp
from test_morphology_runtime import fixture


class MorphologyInterpolationTest(unittest.TestCase):
    def make_lib(self, path, cfg, enabled=True):
        angles = np.array([-.65, .7, -.1], np.float32)
        pose = np.zeros((3,3,3), np.float32)
        pose[:,0] = np.array([[.02,-.1,.0],[.1,.05,.5],[-.1,.1,.7]])
        pose[:,1,0] = -angles
        motion = dict(fps=30, pose_aa=pose, dof_pos=angles[:,None], joint_names=['hinge_joint'],
            root_trans_offset=np.array([[.1,.2,1],[.2,.1,1.02],[.3,.3,1.1]], np.float32),
            metadata=dict(morphology_delta_m=[.04,0,0,0,0], morphology_reference='retargeted', valid_for_training=True))
        motion_file = path.parent/'motions.pkl'
        joblib.dump({'test':motion},motion_file)
        c = EasyDict(motion_file=str(motion_file), asset=dict(assetRoot=str(path.parent),assetFileName=path.name),
            extend_config=[dict(parent_name='length',joint_name='tip',pos=[.1,.02,.03],rot=[1.,0,0,0])],has_mesh=False)
        if enabled: c.morphology=cfg
        lib=MotionLibRobot(c,1,'cpu');lib.load_all_motions()
        return lib

    def test_off_grid_pose_and_velocity_equal_mujoco_qpos_jacobian(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,path,xml=fixture(Path(tmp));lib=self.make_lib(path,cfg)
            times=torch.tensor([.011,.02,.051,.062])
            state=lib.get_motion_state(0,times,offset=torch.tensor([.3,.2,.1]))
            model=mujoco.MjModel.from_xml_string(xml(.04)); data=mujoco.MjData(model)
            for i in range(len(times)):
                data.qpos[:3]=state['root_pos'][i];data.qpos[3:7]=state['root_rot'][i][[3,0,1,2]]
                data.qpos[-1]=state['dof_pos'][i,0]
                data.qvel[:3]=state['root_vel'][i]
                data.qvel[3:6]=Rotation.from_quat(state['root_rot'][i]).inv().apply(state['root_ang_vel'][i])
                data.qvel[-1]=state['dof_vel'][i,0]
                mujoco.mj_forward(model,data)
                np.testing.assert_allclose(state['rg_pos'][i],data.xpos[1:],atol=3e-7)
                rq=Rotation.from_quat(state['rb_rot'][i]).as_matrix()
                np.testing.assert_allclose(rq,data.xmat[1:].reshape(-1,3,3),atol=4e-7)
                for body in range(1,model.nbody):
                    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
                    mujoco.mj_jacBody(model,data,jp,jr,body)
                    np.testing.assert_allclose(state['body_vel'][i,body-1],jp@data.qvel,atol=3e-6)
                    np.testing.assert_allclose(state['body_ang_vel'][i,body-1],jr@data.qvel,atol=3e-6)
                tip=data.xpos[-1]+data.xmat[-1].reshape(3,3)@np.array([.1,.02,.03])
                np.testing.assert_allclose(state['rg_pos_t'][i,-1],tip,atol=3e-7)
            # Positions, including virtual tips, differentiate to returned velocity.
            eps=1e-5
            after=lib.get_motion_state(0,times+eps);before=lib.get_motion_state(0,times-eps)
            finite_difference=(after['rg_pos_t']-before['rg_pos_t'])/(2*eps)
            torch.testing.assert_close(state['body_vel_t'],finite_difference,atol=.008,rtol=.002)
            # Neither original source coordinates nor the parser's nominal shape changed.
            np.testing.assert_allclose(lib.mesh_parsers._offsets[0,2], [0,0,.2])
            np.testing.assert_allclose(lib.dof_pos[:,0],[-.65,.7,-.1],atol=1e-7)

    def test_legacy_path_keeps_independent_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,path,_=fixture(Path(tmp));lib=self.make_lib(path,cfg,enabled=False)
            time=torch.tensor([.02]);state=lib.get_motion_state(0,time)
            expected=.4*lib.gts[0]+.6*lib.gts[1]
            torch.testing.assert_close(state['rg_pos'][0],expected)
            self.assertFalse(hasattr(lib,'_morphology_kinematics'))

    def test_clamped_endpoints_have_zero_derivative(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,path,_=fixture(Path(tmp));lib=self.make_lib(path,cfg)
            state=lib.get_motion_state(0,torch.tensor([-.1,2/30,.5]))
            for key in ['dof_vel','root_vel','root_ang_vel','body_vel','body_ang_vel']:
                torch.testing.assert_close(state[key],torch.zeros_like(state[key]))
            torch.testing.assert_close(state['rg_pos'][1],state['rg_pos'][2])

    def test_quaternion_derivative_matches_near_and_spherical_interpolation(self):
        q0=torch.tensor(Rotation.from_rotvec([[.1,.2,.3],[0,0,0]]).as_quat(),dtype=torch.float64)
        q1=torch.tensor(Rotation.from_rotvec([[.102,.198,.305],[.4,-.2,.6]]).as_quat(),dtype=torch.float64)
        q1[1]*=-1 # shortest arc must be independent of quaternion sign
        blend=torch.tensor([[.27],[.79]],dtype=torch.float64)
        dt=.03
        rot,omega=interpolate_root(q0,q1,blend,torch.ones_like(blend)/dt)
        torch.testing.assert_close(rot,slerp(q0,q1,blend),atol=1e-12,rtol=1e-12)
        eps=1e-5
        before=Rotation.from_quat(slerp(q0,q1,blend-eps));after=Rotation.from_quat(slerp(q0,q1,blend+eps))
        finite=(after*before.inv()).as_rotvec()/(2*eps*dt)
        np.testing.assert_allclose(omega,finite,atol=1e-8)

    def test_reordered_hinges_nonzero_anchor_and_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'robot.xml'
            path.write_text('''<mujoco><compiler angle="radian"/><worldbody><body name="base">
              <freejoint/><geom type="sphere" size=".1" mass="1"/>
              <body name="a" pos=".1 .2 .3" quat=".9238795325 0 0 .3826834324">
                <joint name="a_joint" axis="0 1 0" pos=".02 0 .03" ref=".1"/>
                <geom type="sphere" size=".05" mass="1"/>
                <body name="b" pos="0 .1 .3"><joint name="b_joint" axis="-1 0 0"/>
                <geom type="sphere" size=".03" mass="1"/></body>
              </body></body></worldbody></mujoco>''')
            cfg=dict(family_names=['x'],body_offsets=[dict(body_name='b',family_index=0,axis=[0,0,1])])
            fk=MorphologyKinematics(path,['base','a','b'],['b_joint','a_joint'],cfg,[],'cpu')
            model=mujoco.MjModel.from_xml_path(str(path)); model.body_pos[3,2]+=.04;model.body_sameframe[:]=0
            data=mujoco.MjData(model)
            pos=torch.tensor([[.1,-.2,1.]])
            rot=torch.tensor(Rotation.from_rotvec([.1,.2,-.3]).as_quat()[None],dtype=torch.float32)
            vel=torch.tensor([[.1,.2,-.3]]);omega=torch.tensor([[.4,-.2,.1]])
            qp=torch.tensor([[-.3,.4]]);qv=torch.tensor([[.2,-.7]])
            p,r,v,w=fk.forward(pos,rot,vel,omega,qp,qv,torch.tensor([[.04]]))
            data.qpos[:3]=pos[0];data.qpos[3:7]=rot[0,[3,0,1,2]];data.qpos[7:]=qp[0,[1,0]]
            data.qvel[:3]=vel[0];data.qvel[3:6]=Rotation.from_quat(rot[0]).inv().apply(omega[0]);data.qvel[6:]=qv[0,[1,0]]
            mujoco.mj_forward(model,data)
            np.testing.assert_allclose(p[0],data.xpos[1:],atol=3e-7)
            for body in range(1,4):
                jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv));mujoco.mj_jacBody(model,data,jp,jr,body)
                np.testing.assert_allclose(v[0,body-1],jp@data.qvel,atol=3e-7)
                np.testing.assert_allclose(w[0,body-1],jr@data.qvel,atol=3e-7)
            with self.assertRaisesRegex(ValueError,'explicit joint_names'):
                MorphologyKinematics(path,['base','a','b'],None,cfg,[],'cpu')


if __name__=='__main__':unittest.main()
