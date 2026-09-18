#!/usr/bin/env bash
# Train and score the whole model family, one size after another.
#
#   ./run_pipeline.sh data                                  # rebuild npy + tfrecords
#   ./run_pipeline.sh sweep                                 # all four sizes, end to end
#   ./run_pipeline.sh sweep resumamba_30k                   # ... or just this one
#   ./run_pipeline.sh summary                               # the comparison table again
#
# Each size runs ssl -> cpc -> train -> stepeval -> ec57. Both self-supervised stages are
# label-free and depend only on the architecture, so they are skipped when weights already
# exist - in this run, or in the run ECGR_SSL_RUN / ECGR_CPC_RUN names.
#
# A sweep runs for hours, so START IT DETACHED - `nohup ... &` alone is not enough, it stays
# in the launching shell's process group and dies with it:
#
#   setsid nohup ./run_pipeline.sh sweep </dev/null >logs/sweep.log 2>&1 &
#
# Two free cards = two queues, one per card. Each model writes under its own
# <CHECKPOINT_DIR>/<model>/ and each EC57 into <EC57_DIR>/<tag>/ with its own symlink farm,
# so parallel queues cannot collide. Pin ECGR_RUN_TAG so both write the same run:
#
#   export ECGR_RUN_TAG=260917_3lead
#   ECGR_GPU=0 setsid nohup ./run_pipeline.sh sweep resumamba_2m resumamba_100k </dev/null >logs/q0.log 2>&1 &
#   ECGR_GPU=1 setsid nohup ./run_pipeline.sh sweep resumamba_1m resumamba_30k  </dev/null >logs/q1.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")"
# The conda env this project is developed against; override with PY=/path/to/python.
PY="${PY:-/home/ai-server/miniconda3/envs/beat/bin/python}"
[[ -x "${PY}" ]] || { echo "python not found: ${PY} (set PY=...)" >&2; exit 2; }

export ECGR_RUN_TAG="${ECGR_RUN_TAG:-$(date +%y%m%d)_3lead}"
export ECGR_IN_CHANNELS="${ECGR_IN_CHANNELS:-3}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export CUDA_VISIBLE_DEVICES="${ECGR_GPU:-0}"

GPU="${ECGR_GPU:-0}"
GPU_FREE_MIB="${ECGR_GPU_FREE_MIB:-9000}"          # wait for this much VRAM before training
GPU_FREE_INFER_MIB="${ECGR_GPU_FREE_INFER_MIB:-4000}"
GPU_WAIT_MAX="${ECGR_GPU_WAIT_MAX:-21600}"         # ... but no longer than this (s); 0 = never wait

# Largest first: the big sizes are the ones worth a free card early, and a queue that dies
# halfway has then produced the results that matter most.
SIZES=(resumamba_2m resumamba_1m resumamba_100k resumamba_30k)

LOGS="$PWD/logs"
mkdir -p "$LOGS"

# Every stage waits, not just training: a step eval or an EC57 sweep that starts on a full
# card dies with an allocator error, and with one shared card that is the likeliest way an
# unattended run loses a stage.
wait_gpu() {
    local need="${1:-${GPU_FREE_MIB}}" waited=0 free
    [[ "${GPU_WAIT_MAX}" == "0" ]] && return 0
    while :; do
        free=$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits)
        [[ "${free}" -ge "${need}" ]] && { echo "    GPU${GPU}: ${free} MiB free, starting"; return 0; }
        if [[ ${waited} -ge ${GPU_WAIT_MAX} ]]; then
            echo "    GPU${GPU}: only ${free} MiB free after ${waited}s, starting anyway" >&2
            return 0
        fi
        [[ $((waited % 600)) -eq 0 ]] && echo "    GPU${GPU}: ${free} MiB free < ${need}, waiting (${waited}s)"
        sleep 60; waited=$((waited + 60))
    done
}

# tee the whole log, show only the lines worth watching - but take the exit status from
# PIPESTATUS[0], not from grep: "|| true" on the pipeline would turn a crashed stage into a
# success and the next stage would then score whatever stale checkpoint it found.
run_stage() {
    local log="$1" filter="$2"; shift 2
    set +e
    "$@" 2>&1 | tee "${log}" | grep -E "${filter}"
    local rc=${PIPESTATUS[0]}
    set -e
    [[ ${rc} -eq 0 ]] || { echo "FAILED (rc=${rc}), full log: ${log}" >&2; return ${rc}; }
}

stage_ssl() {
    wait_gpu "${GPU_FREE_MIB}"
    echo "[$(date +%H:%M:%S)] ssl $1 (self-supervised backbone, labels unused)"
    run_stage "${LOGS}/$1_ssl.log" \
        "^(Epoch [0-9]|backbone|masking|data |SSL-pretrained|nmse|Traceback|.*Error)" \
        "${PY}" -m ecgr ssl --model "$1"
}

stage_cpc() {
    wait_gpu "${GPU_FREE_MIB}"
    echo "[$(date +%H:%M:%S)] cpc $1 (self-supervised context encoder, labels unused)"
    run_stage "${LOGS}/$1_cpc.log" \
        "^(Epoch [0-9]|context encoder|negatives|data |CPC-pretrained|top1|Traceback|.*Error)" \
        "${PY}" -m ecgr cpc --model "$1"
}

stage_train() {
    wait_gpu "${GPU_FREE_MIB}"
    echo "[$(date +%H:%M:%S)] train $1"
    run_stage "${LOGS}/$1_train.log" \
        "^(Epoch [0-9]|model +:|monitor +:|backbone +:|context_encoder *:|data +:|checkpoints|new best|measured only|Weighted F1|Traceback|.*Error)" \
        "${PY}" -m ecgr train --model "$1" "${@:2}"
}

stage_stepeval() {
    wait_gpu "${GPU_FREE_INFER_MIB}"
    echo "[$(date +%H:%M:%S)] stepeval $1"
    run_stage "${LOGS}/$1_stepeval.log" \
        "(Weighted F1|^ *(None|N|V|S) |report ->|checkpoint:|Traceback|.*Error)" \
        "${PY}" -m ecgr stepeval --model "$1"
}

stage_ec57() {
    wait_gpu "${GPU_FREE_INFER_MIB}"
    echo "[$(date +%H:%M:%S)] ec57 $1 (physionet, one lead duplicated + portal 3-lead)"
    run_stage "${LOGS}/$1_ec57.log" \
        "(^=====|^EC57 report|^  (Average|Gross|Total)|^EC57 summary|^  db=|Traceback|.*Error)" \
        "${PY}" -m ecgr ec57 --model "$1"
}

STAGE="${1:?stage: data|sweep|summary|test}"
shift || true

echo "=== ${STAGE} ==="
"${PY}" -m ecgr config
echo "    gpu : ${GPU} (train needs ${GPU_FREE_MIB} MiB free, inference ${GPU_FREE_INFER_MIB})"

case "${STAGE}" in
  data)
    "${PY}" -m ecgr data --step all "$@"
    ;;

  test)
    "${PY}" -m pytest tests/ -q "$@"
    ;;

  sweep)
    if [[ $# -gt 0 ]]; then list=("$@"); else list=("${SIZES[@]}"); fi

    # Every size is isolated: one that fails is recorded and the queue moves on, because
    # letting `set -e` abort here would throw away the sizes that had nothing wrong with them.
    failures=()
    CKPT_ROOT="$("${PY}" -c "from ecgr import config; print(config.CHECKPOINT_DIR)")"
    for m in "${list[@]}"; do
        keras="resumamba_seq2seq_${m#resumamba_}"
        # The self-supervised stages are architecture-only, so rerunning them would just
        # spend GPU time reproducing the same weights.
        if [[ -f "$("${PY}" -c "from ecgr import config; print(config.ssl_weights_dir('${keras}'))")" ]]; then
            echo "[$(date +%H:%M:%S)] ${m}: backbone present, skipping ssl"
        elif ! stage_ssl "$m"; then
            failures+=("${m}:ssl"); continue
        fi
        if [[ -f "$("${PY}" -c "from ecgr import config; print(config.cpc_weights_dir('${keras}'))")" ]]; then
            echo "[$(date +%H:%M:%S)] ${m}: context encoder present, skipping cpc"
        elif ! stage_cpc "$m"; then
            failures+=("${m}:cpc"); continue
        fi
        if ls "${CKPT_ROOT}/${keras}/BEST_F1/"*.keras >/dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] ${m}: checkpoint present, skipping train"
        elif ! stage_train "$m"; then
            failures+=("${m}:train"); continue
        fi
        stage_stepeval "$m" || failures+=("${m}:stepeval")
        stage_ec57 "$m"     || failures+=("${m}:ec57")
    done

    "${PY}" -m ecgr compare "${list[@]}" || true
    if [[ ${#failures[@]} -gt 0 ]]; then
        echo "=== ${STAGE} FINISHED WITH FAILURES: ${failures[*]} ==="
        echo "    rerun the same command - finished stages are skipped, only failed ones redo"
        exit 1
    fi
    echo "=== ${STAGE} complete ($(date +%H:%M:%S)) ==="
    ;;

  summary)
    "${PY}" -m ecgr compare "${@:-${SIZES[@]}}"
    ;;

  *) echo "unknown stage ${STAGE}" >&2; exit 2 ;;
esac
