"""CPU configuration and saved-clock accounting only; never construct a model or env."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from humanoidverse.train import build_ufo_mjlab_config, parse_args
from humanoidverse.training.checkpoint_io import ReplayWarmup
from humanoidverse.agents.utils import EveryNStepsChecker


ROBOT = Path(__file__).resolve().parents[1]/"configs/robots/g1_29dof.yaml"


class NumSeedStepsTest(unittest.TestCase):
    def test_cli_default_explicit_and_reject_nonpositive(self):
        for flags,expected in [((),None),(('--num-seed-steps','1024000'),1024000)]:
            with patch.object(sys,'argv',['train.py','--robot-config',str(ROBOT),*flags]):
                self.assertEqual(parse_args().num_seed_steps,expected)
        for value in ['0','-1','1.5']:
            with patch.object(sys,'argv',['train.py','--robot-config',str(ROBOT),'--num-seed-steps',value]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError,SystemExit)):parse_args()
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_config_default_unchanged_and_only_seed_override(self):
        args=dict(device='cpu',work_dir='/tmp/config_only',num_envs=1,num_env_steps=1,seed=1,
                  use_wandb=False,wandb_run_name=None,smoke=True,compile_agent=False,robot_config=ROBOT)
        old=build_ufo_mjlab_config(**args).model_dump()
        omitted=build_ufo_mjlab_config(**args,num_seed_steps=None).model_dump()
        self.assertEqual(old,omitted);self.assertEqual(old['num_seed_steps'],10240)
        new=build_ufo_mjlab_config(**args,num_seed_steps=1024000).model_dump()
        self.assertEqual({**old,'num_seed_steps':1024000},new)
        for value in [0,-1,True,1.5,'1024000']:
            with self.assertRaises(ValueError):build_ufo_mjlab_config(**args,num_seed_steps=value)
        self.assertFalse(torch.cuda.is_initialized())

    def test_resumed_cold_replay_keeps_saved_clocks_and_952_updates(self):
        for seed_steps,endpoint,controls,first_update in [(10240,32555008,128,10),(1024000,40665088,1118,1000)]:
            local=31506432//8;updates=60640
            warmup=ReplayWarmup.for_resume(transactional=True,resumed=True,replay_loaded=False,seed_steps=seed_steps)
            checker=EveryNStepsChecker(local,1024);triggered=[]
            for i in range(1,controls+1):
                # Existing Workspace advances replay before checking its original
                # absolute local clock and update trigger; no zeroing saved clocks.
                warmup.advance(1024)
                if warmup.ready and local>seed_steps and checker.check(local):
                    checker.update_last_step(local);updates+=8;triggered.append(i)
                local+=1024
            self.assertEqual(local*8,endpoint)
            self.assertEqual((triggered[0],len(triggered),updates),(first_update,119,61592))
            self.assertEqual(warmup.collected_steps,controls*1024)


if __name__=='__main__':unittest.main()
