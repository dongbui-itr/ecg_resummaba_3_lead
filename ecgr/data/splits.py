"""Who may be trained on, and which side of the train/eval line they fall.

The single most expensive mistake this project can make is training on a study that is later
scored, because every number downstream then looks good and means nothing. So the holdout is
enforced three times, at increasing distance from the code that computes it:

  1. held_out_studies()     - drops the test/eval studies before the split even happens
  2. verify_split_integrity - proves the disjointness BEFORE any file is written
  3. audit_written_data     - re-proves it AFTER the build, from the study ids stored inside
                              the npy batches, so it cannot be fooled by a build that wrote
                              something other than what it computed
"""
import glob
import hashlib
import json
import os
import re
from datetime import datetime

import pandas as pd

from .. import config


def study_split_side(study_id, train_fraction=None):
    """Which split a study belongs to, decided by a hash of its id.

    The same study appears in several portal datasets (they share the portal), so the split
    must be a GLOBAL property of the study - independent per-dataset splits put thousands of
    studies in the train split of one dataset and the eval split of another. A deterministic
    hash gives every study the same side in every dataset, run and machine, with no shared
    state to keep in sync.
    """
    fraction = config.TRAIN_FRACTION if train_fraction is None else train_fraction
    digest = hashlib.md5(str(study_id).strip().encode()).hexdigest()
    return 'train' if int(digest, 16) % 100 < int(fraction * 100) else 'eval'


def split_studies(study_ids):
    train = [s for s in study_ids if study_split_side(s) == 'train']
    evaluation = [s for s in study_ids if study_split_side(s) == 'eval']
    return train, evaluation


def eval_studies():
    """Study ids reserved for the v4 eval sets (config.EVAL_STUDIES_JSON).

    Missing the file is an error, not a warning: silently building training data on top of
    the eval studies is exactly what this guard exists to prevent.
    """
    if not config.EXCLUDE_EVAL_STUDIES:
        print("EXCLUDE_EVAL_STUDIES is off - the eval studies are NOT held out")
        return set()
    path = config.EVAL_STUDIES_JSON
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"eval study list not found: {path}. Point ECGR_EVAL_STUDIES_JSON at it, or set "
            f"config.EXCLUDE_EVAL_STUDIES = False to build data without the holdout.")
    with open(path) as f:
        ids = {str(s).strip() for s in json.load(f)['eval_studyids']}
    print(f"eval holdout: {len(ids)} studies from {path}")
    return ids


_RECORD_NAME = re.compile(r'^(\d+)_[0-9a-fA-F]{24}_')


def _test_studies_of(name):
    """Study ids of one test dataset, from whichever inventory that folder actually has.

    Three shapes exist under config.DATA_DIR:
      1. dataset_info_full.csv     - the normal portal layout
      2. list_studies_*.json       - a scanned study list, which covers BOTH the beat and the
                                     rhythm eval sets, not only the records in one folder
      3. <study>_<event>_*.dat     - last resort: read the ids off the file names
    Preferring the JSON over the file names matters: dataset-eval/v4 holds the beat records
    only, but its JSON lists the rhythm eval studies too, and those must not be trained on.
    """
    root = os.path.join(config.DATA_DIR, name)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"test dataset '{name}' not found: {root}")

    csv_path = os.path.join(root, config.DATASET_CSV)
    if os.path.exists(csv_path):
        ids = {str(s).strip() for s in pd.read_csv(csv_path)['study_id'].unique()}
        return ids, f"{name}/{config.DATASET_CSV}"

    json_paths = [p for p in sorted(glob.glob(os.path.join(root, '**', 'list_studies_*.json'),
                                              recursive=True))
                  if not p.endswith('_detail.json')]
    if json_paths:
        ids = set()
        for path in json_paths:
            with open(path) as f:
                info = json.load(f)
            for key in ('eval_studyids', 'study_ids', 'studyids'):
                if key in info:
                    ids |= {str(s).strip() for s in info[key]}
                    break
        if ids:
            return ids, ', '.join(os.path.relpath(p, config.DATA_DIR) for p in json_paths)

    ids = {m.group(1) for _, _, files in os.walk(root) for f in files
           if (m := _RECORD_NAME.match(f))}
    if ids:
        return ids, f"{name}/ (record file names)"

    raise FileNotFoundError(
        f"test dataset '{name}': no {config.DATASET_CSV}, no list_studies_*.json and no "
        f"<study>_<event>_* records under {root} - cannot tell which studies to hold out")


def test_studies():
    """Every study_id of the test datasets (config.TEST_DATASETS), as strings."""
    ids = set()
    for name in config.TEST_DATASETS:
        found, source = _test_studies_of(name)
        print(f"test holdout: {len(found)} studies from {source}")
        ids |= found
    if not ids:
        raise ValueError("config.TEST_DATASETS is empty - refusing to build training data "
                         "with no test holdout")
    return ids


def held_out_studies():
    """Union of the test datasets and the v4 eval list: nothing here may be trained on."""
    tests = test_studies()
    return tests | eval_studies(), tests


def verify_split_integrity(db_name, train_ids, eval_ids, test_ids=None, out_path=None):
    """Hard guarantee that no study id crosses a split boundary.

    Raises ValueError on the first violation BEFORE any data is written. Checks:
      * train and eval are disjoint
      * neither contains a test study
      * cross-dataset: a study is never train here and eval in another dataset, compared
        against the verification files earlier datasets of this rebuild already wrote
    The verified lists and every intersection count go to out_path so the split stays
    auditable; on a violation the file is still written (verified: false) before raising.
    """
    test = {int(s) for s in (test_ids or set())}
    train = {int(s) for s in train_ids}
    evaluation = {int(s) for s in eval_ids}

    overlaps = {
        'train_x_eval': sorted(train & evaluation),
        'train_x_test_datasets': sorted(train & test),
        'eval_x_test_datasets': sorted(evaluation & test),
    }
    for other_file in glob.glob(os.path.join(config.NPY_DIR, '*', 'split_verification.json')):
        other_db = os.path.basename(os.path.dirname(other_file))
        if other_db == db_name:
            continue
        with open(other_file) as f:
            other = json.load(f)
        overlaps[f'train_x_{other_db}_eval'] = sorted(train & set(other['eval_studyids']))
        overlaps[f'eval_x_{other_db}_train'] = sorted(evaluation & set(other['train_studyids']))

    violations = {k: v for k, v in overlaps.items() if v}

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'db_name': db_name,
                'verified': not violations,
                'checked_at': datetime.now().isoformat(timespec='seconds'),
                'test_datasets': list(config.TEST_DATASETS),
                'counts': {'train': len(train), 'eval': len(evaluation), 'test': len(test)},
                'overlap_counts': {k: len(v) for k, v in overlaps.items()},
                'overlap_studyids': violations,          # empty when verified
                'train_studyids': sorted(train),
                'eval_studyids': sorted(evaluation),
                'test_studyids': sorted(test),
            }, f, indent=4)

    if violations:
        detail = ', '.join(f"{k}: {len(v)} studies (e.g. {v[:5]})" for k, v in violations.items())
        raise ValueError(f"{db_name}: split integrity violated - {detail}. "
                         f"No data was written; fix the split before rebuilding.")
    print(f"{db_name}: split integrity OK "
          f"(train n eval = train n test = eval n test = 0)")


def audit_written_data(db_names=None, out_path=None):
    """Re-prove the separation from the study ids stored IN the npy batches.

    Independent of how the split was computed: it reads back what was actually written
    (<db>_<split>_studyids_batch_*.npy) and the test studies straight from the test dataset
    inventories, then checks, over all datasets pooled:

        train n eval = 0,  train n test = 0,  eval n test = 0
    """
    import numpy as np
    db_names = db_names or config.TRAIN_DATASETS
    written = {'train': set(), 'eval': set()}
    for db in db_names:
        for split in written:
            for path in glob.glob(os.path.join(config.NPY_DIR, db, split,
                                               f"{db}_{split}_studyids_batch_*.npy")):
                written[split] |= {int(v) for v in np.load(path)}

    test = {int(s) for s in test_studies()}
    overlaps = {
        'train_x_eval': sorted(written['train'] & written['eval']),
        'train_x_test': sorted(written['train'] & test),
        'eval_x_test': sorted(written['eval'] & test),
    }
    violations = {k: v for k, v in overlaps.items() if v}
    report = {
        'checked_at': datetime.now().isoformat(timespec='seconds'),
        'datasets': list(db_names),
        'verified': not violations,
        'counts': {k: len(v) for k, v in written.items()} | {'test': len(test)},
        'overlap_counts': {k: len(v) for k, v in overlaps.items()},
        'overlap_studyids': violations,
    }
    out_path = out_path or os.path.join(config.NPY_DIR, 'audit_written_data.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"audit: {report['counts']} -> {out_path}")
    if violations:
        raise ValueError(f"written data violates the split: "
                         f"{ {k: len(v) for k, v in violations.items()} }")
    print("audit OK: train n eval = train n test = eval n test = 0")
    return report
