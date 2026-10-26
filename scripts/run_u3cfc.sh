#!/usr/bin/env bash
# Train rhythm_u3cfc_1m from scratch (three-U-Net encoder <= 900K, CNN + CfC decoder <= 300K),
# then score it exactly like the dualcfc baseline it has to match:
#
#   beat   : bxb    (scripts/dualcfc_bxb.py)  - EC57 physionet dbs + v4 portal beat-eval,
#            twice: as-is, and with the noise gate (no beat in a 2 s segment with
#            p(NOISE) >= 0.5) - the nstdb operating point of the brief
#   rhythm : epicmp (ecgr.rhythm ec57)        - mitdb/afdb/escdb + the rhythm holdout
#
# Data: npy_rhythm_141026, the dualcfc build - no EC57 database among its sources, 0 studies
# in common with the v4 list or the v4 records on disk (checked 2026-10-20).
#
#   ECGR_GPU=0 setsid nohup scripts/run_u3cfc.sh </dev/null >logs/q_u3cfc.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/rhythm_env_local.sh
export ECGR_RHYTHM_NPY_DIR=/mnt/md0/Dong_data/portal_data/npy_rhythm_141026
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES="${ECGR_GPU:-0}"
# noise label in nstdb's own SNR (QRS p-p), NOISE below 6 dB; levels drawn over nstdb's range
export ECGR_RHYTHM_NOISE_SNR_REF=qrs_pp
export ECGR_RHYTHM_NOISE_LABEL_SNR_DB=6
export ECGR_RHYTHM_NOISE_SNR_MIN=-12
export ECGR_RHYTHM_NOISE_SNR_MAX=24
# electrode-motion noise in the generator (augment._electrode_motion): without it the noise head
# never sees the 0.5-5 Hz wobble that real motion artefact is, and flags none of it on nstdb
export ECGR_RHYTHM_NOISE_EM="${ECGR_RHYTHM_NOISE_EM:-1}"
# keep the epoch best on beat AND rhythm, not the rhythm peak alone
export ECGR_RHYTHM_MONITOR=val_joint
PY=/home/ai-server/miniconda3/envs/beat/bin/python
M="${MODEL:-rhythm_u3cfc_1m}"
KN="u3cfc_rhythm_${M#rhythm_u3cfc_}"
TAG="${ECGR_U3CFC_TAG:-201026_u3cfc_1m_em}"
CKPT=/mnt/md0/Dong_data/portal_data/train/${TAG}/checkpoints/${KN}/best_model.keras
LOGD=logs/u3cfc/${TAG}
mkdir -p "${LOGD}"

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
    echo "[$(date +%H:%M:%S)] train ${M} -> ${TAG}"
    ECGR_RHYTHM_RUN_TAG="${TAG}" "${PY}" -m ecgr.rhythm train --model "${M}" \
        --schedule cosine --sampler stratified --epochs "${EPOCHS:-40}" --patience 8 \
        ${INIT_FROM:+--init-from "${INIT_FROM}"} ${LR:+--lr "${LR}"} \
        ${INIT_MATCHING:+--init-matching "${INIT_MATCHING}"} \
        ${FREEZE:+--freeze-epochs "${FREEZE}"} ${LR2:+--lr2 "${LR2}"} \
        > "${LOGD}/train.log" 2>&1
fi
[ -f "${CKPT}" ] || { echo "[$(date +%H:%M:%S)] MISSING ${CKPT}"; exit 1; }

echo "[$(date +%H:%M:%S)] bxb (no gate)"
ECGR_RUN_TAG="${TAG}_eval" "${PY}" scripts/dualcfc_bxb.py --checkpoint "${CKPT}" \
    --tag u3cfc_1m > ${LOGD}/bxb.log 2>&1
echo "[$(date +%H:%M:%S)] bxb (noise gate 0.5)"
ECGR_RUN_TAG="${TAG}_eval" "${PY}" scripts/dualcfc_bxb.py --checkpoint "${CKPT}" \
    --tag u3cfc_1m_gate05 --noise-gate 0.5 > ${LOGD}/bxb_gate05.log 2>&1
echo "[$(date +%H:%M:%S)] epicmp (physionet + rhythm holdout)"
ECGR_RHYTHM_RUN_TAG="${TAG}" "${PY}" -m ecgr.rhythm ec57 --checkpoint "${CKPT}" \
    --tag u3cfc_1m > ${LOGD}/epicmp.log 2>&1
echo "[$(date +%H:%M:%S)] nstdb noise vs true SNR"
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda "${PY}" scripts/u3cfc_noise_report.py \
    --checkpoint "${CKPT}" > ${LOGD}/noise_report.log 2>&1
echo "[$(date +%H:%M:%S)] all done"
