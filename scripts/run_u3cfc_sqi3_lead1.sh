#!/usr/bin/env bash
# rhythm_u3cfc_1m_sqi3 (same as sqi2 + beat types into the noise path): train ONLY the new noise path (full ratio + TP ratio + TP coverage) on top
# of the frozen rhythm_u3cfc_1m_sqi best checkpoint, then score noise + gated beats.
# Beat and rhythm weights stay exactly those of the sqi checkpoint, so its bxb (ungated) and
# epicmp results stand unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/rhythm_env_local.sh
export ECGR_RHYTHM_NPY_DIR=/mnt/md0/Dong_data/portal_data/npy_rhythm_141026
export TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES="${ECGR_GPU:-0}"
export ECGR_RHYTHM_NOISE_SNR_REF=qrs_pp ECGR_RHYTHM_NOISE_LABEL_SNR_DB=6
export ECGR_RHYTHM_NOISE_SNR_MIN=-12 ECGR_RHYTHM_NOISE_SNR_MAX=24 ECGR_RHYTHM_NOISE_EM=1
export ECGR_RHYTHM_NOISE_LABEL_MIN_LEADS=1
export ECGR_RHYTHM_MONITOR=val_joint ECGR_RHYTHM_CKPT_START_EPOCH=1
PY=/home/ai-server/miniconda3/envs/beat/bin/python
TAG=241026_u3cfc_1m_sqi3_lead1
SRC=/mnt/md0/Dong_data/portal_data/train/241026_u3cfc_1m_sqi3_lead1/lineage/231026_u3cfc_1m_sqi/best_model.keras   # the sqi checkpoint, kept in the final model's lineage/
LOGD=logs/u3cfc/${TAG}; mkdir -p "${LOGD}"
CKD=/mnt/md0/Dong_data/portal_data/train/${TAG}/checkpoints/u3cfc_rhythm_1m_sqi3/epochs
echo "[$(date +%H:%M:%S)] train noise path -> ${TAG}"
ECGR_RHYTHM_RUN_TAG="${TAG}" "${PY}" -m ecgr.rhythm train --model rhythm_u3cfc_1m_sqi3 \
    --schedule cosine --sampler stratified --epochs 6 --freeze-epochs 6 --lr 1e-3 \
    --init-matching "${SRC}" > "${LOGD}/train.log" 2>&1
CKPT=${CKD}/epoch_06.keras          # frozen beat/rhythm => val_joint flat; the last epoch (6) is the model = final_model.keras
echo "[$(date +%H:%M:%S)] nstdb noise vs true SNR"
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda "${PY}" scripts/u3cfc_noise_report.py \
    --checkpoint "${CKPT}" > "${LOGD}/noise_report.log" 2>&1
echo "[$(date +%H:%M:%S)] bxb (noise gate 0.2, the operating point)"
ECGR_RUN_TAG="${TAG}_eval" "${PY}" scripts/dualcfc_bxb.py --checkpoint "${CKPT}" \
    --tag final_gate0.2 --noise-gate 0.2 > "${LOGD}/bxb_final_gate02.log" 2>&1
echo "[$(date +%H:%M:%S)] all done"
