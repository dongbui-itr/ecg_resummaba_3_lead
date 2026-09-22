"""Data separation, checked against what is on disk rather than against what was intended.

The most expensive mistake this project can make is scoring a study it trained on: every
number downstream then looks good and means nothing. splits.py already enforces the holdout
three times during the build; these tests re-prove it AFTER the fact, from the study ids
stored inside the npy batches and from the resolved tfrecord paths, and they add the one
check the build cannot make - that the split used to SELECT checkpoints is disjoint from the
sets used to REPORT them.

They skip when the data is not on this machine, so the suite still runs anywhere.
"""
import glob
import os
import re

import numpy as np
import pytest

from ecgr import config

TFRECORDS = os.path.isdir(config.TFRECORD_DIR)
NPY = os.path.isdir(config.NPY_DIR)
V4 = config.PORTAL_EVAL_SETS['dataset-v4-beat']

needs_tfrecords = pytest.mark.skipif(not TFRECORDS, reason='tfrecords not built here')
needs_npy = pytest.mark.skipif(not NPY, reason='npy tree not built here')
needs_v4 = pytest.mark.skipif(not os.path.isdir(V4), reason='v4 beat-eval set absent')


def written_study_ids(split):
    """Study ids actually stored in the npy batches of `split`, over all training datasets."""
    ids = set()
    for db in config.TRAIN_DATASETS:
        for path in glob.glob(os.path.join(config.NPY_DIR, db, split,
                                           f"{db}_{split}_studyids_batch_*.npy")):
            ids |= {int(v) for v in np.load(path)}
    return ids


def v4_study_ids():
    """Study ids of the beat-eval records, read off the record file names."""
    names = {f[:-4] for f in os.listdir(V4) if f.endswith('.dat')}
    return {int(m.group(1)) for n in names if (m := re.match(r'^(\d+)_', n))}


# --- the EC57 benchmark must not be in the training data at all ------------------------

@needs_tfrecords
def test_no_training_tfrecord_comes_from_an_ec57_database():
    """mitdb / nstdb / escdb / ahadb / afdb are the independent benchmark. One tfrecord built
    from them would turn every EC57 number into self-scoring."""
    from ecgr.data import pipeline
    files = pipeline.split_files('train') + pipeline.split_files('eval')
    physionet = os.path.realpath(config.PHYSIONET_DIR)
    assert not [f for f in files if os.path.realpath(f).startswith(physionet)]
    assert not [f for f in files
                if any(f"/{db}/" in f.lower() for db in config.EC57_DBS)]
    pipeline.assert_no_benchmark_data(files)      # the guard the training path itself uses


def test_the_benchmark_guard_actually_fires():
    """A guard that never says no is not a guard."""
    from ecgr.data import pipeline
    with pytest.raises(RuntimeError, match='EC57'):
        pipeline.assert_no_benchmark_data([os.path.join(config.PHYSIONET_DIR,
                                                        'mitdb/train/x.tfrecord')])
    with pytest.raises(RuntimeError, match='EC57'):
        pipeline.assert_no_benchmark_data(['/somewhere/nstdb/train/x.tfrecord'])


# --- the v4 beat-eval studies must be in neither split --------------------------------

@needs_npy
@needs_v4
def test_no_v4_study_was_written_into_either_split():
    from ecgr.data import splits
    held, _ = splits.held_out_studies()
    held = {int(s) for s in held if str(s).strip().isdigit()}
    v4 = v4_study_ids()
    assert v4, "the beat-eval record names carry no study id - the check would be vacuous"
    assert v4 <= held, f"{len(v4 - held)} beat-eval studies are not in the holdout list"
    for split in ('train', 'eval'):
        written = written_study_ids(split)
        assert written, f"no study ids found in the {split} npy batches"
        assert not (written & v4), f"{split} contains beat-eval studies"
        assert not (written & held), f"{split} contains held-out studies"


@needs_npy
def test_the_two_training_splits_are_disjoint():
    train, evaluation = written_study_ids('train'), written_study_ids('eval')
    assert train and evaluation
    assert not (train & evaluation)


# --- selection set vs reporting sets --------------------------------------------------

@needs_npy
@needs_v4
def test_the_split_used_for_selection_is_disjoint_from_the_one_reported_on():
    """Checkpoints, refinement epochs and s_boost are all chosen on `portal-eval`; the
    numbers are reported on `dataset-v4-beat` and the EC57 databases. If those shared a
    study, selection would be scoring itself - so this is the check that makes the
    no-regression rule in training/select.py and training/refine.py mean anything.
    """
    from ecgr.evaluation import ec57
    selection = {int(sid) for _, sid, *_ in ec57.portal_split_records('eval')}
    assert selection, "portal-eval is empty - the selection rule would be vacuous"
    assert not (selection & v4_study_ids()), "selection and beat-eval share studies"
    assert not (selection & written_study_ids('train')), "selection set is in the train data"
