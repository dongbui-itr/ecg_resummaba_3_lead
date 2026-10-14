# Local paths for the rhythm task on this machine. Source it before any ecgr.rhythm command:
#
#   . scripts/rhythm_env_local.sh && python -m ecgr.rhythm ...
#
# The defaults in ecgr/rhythm/config.py point at /media/Project and /media/MegaDataSet, which
# are the reference machine's mounts and do not exist here. Only the roots differ; nothing
# about the split or the holdout changes with them.
export ECGR_RHYTHM_DATA_ROOT=/mnt/md0/Dong_data/portal_data
# The rhythm test set. This export keys its folders by EVENT id, not study id - resolved
# through ECGR_PORTAL_EVENT_INDEX. Do not point this at anything else: it is the holdout.
export ECGR_RHYTHM_EVAL_DIR=/mnt/md0/Dong_data/portal_data/dataset-v4-rhythm-eval/rhythm-eval-dataset-2
export ECGR_PORTAL_EVENT_INDEX=/mnt/md0/Dong_data/portal_data/dataset_info_full_all.csv
export ECGR_PHYSIONET_TRAIN_DIR=/mnt/md0/Dong_data/physionet_val
export ECGR_RHYTHM_WORK_DIR=/mnt/md0/Dong_data/portal_data/train
# The beat task's config, read for the shared v4 holdout list and the EC57 databases.
export ECGR_DATA_DIR=/mnt/md0/Dong_data/portal_data/
export ECGR_PHYSIONET_DIR=/mnt/md0/Dong_data/physionet/
