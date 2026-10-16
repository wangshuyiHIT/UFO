"""Explicit new CLI. Existing train.launch/old CLI remain unchanged."""
import argparse
from pathlib import Path
from scaler_ufo_adapter.freeze_training_entry import validate_entry
from humanoidverse import train


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--entry',required=True);parser.add_argument('--entry-sha256',required=True)
    ns=parser.parse_args(argv);entry,context,args=validate_entry({'path':ns.entry,'sha256':ns.entry_sha256},execute=True)
    # torchrun supplies the actual rank environment; run_train constructs its own
    # exact projected cfg and verifies it before original Workspace/load/update.
    train.run_train(args,Path(context['work_dir']),obs_normalizer_policy_binding=entry['policy'])
    return 0

if __name__=='__main__':raise SystemExit(main())
