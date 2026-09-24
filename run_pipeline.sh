#!/usr/bin/env bash
# Train and score the whole model family, one size after another.
#
#   ./run_pipeline.sh data                                  # rebuild npy + tfrecords (60 s windows)
#   ./run_pipeline.sh sweep                                 # all four sizes, end to end
#   ./run_pipeline.sh sweep resumamba_100k                  # ... or just this one
#   ./run_pipeline.sh summary                               # the comparison table again
#   ./run_pipeline.sh regress                               # every size against its 10 s baseline
#   ./run_pipeline.sh refine resumamba_5m                   # temporal head on the trained base,
#                                                           #   then EC57 under <model>_refined
#
# Each size runs ssl -> cpc -> train -> select -> stepeval -> ec57 -> regress. Both
# self-supervised stages are label-free and depend only on the architecture, so they are
# skipped when weights already exist - in this run, or in the run ECGR_SSL_RUN / ECGR_CPC_RUN
# names. `select` scores every saved epoch with bxb on portal-eval and picks the beat-level
# winner (step F1 does not pick it - README section 8); ec57 then scores that checkpoint.
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
#   export ECGR_RUN_TAG=260923_60s
#   ECGR_GPU=0 setsid nohup ./run_pipeline.sh sweep resumamba_5m resumamba_100k </dev/null >logs/q0.log 2>&1 &
#   ECGR_GPU=1 setsid nohup ./run_pipeline.sh sweep resumamba_3m resumamba_1m   </dev/null >logs/q1.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")"
# The conda env this project is developed against; override with PY=/path/to/python.
PY="${PY:-/home/ai-server/miniconda3/envs/beat/bin/python}"
[[ -x "${PY}" ]] || { echo "python not found: ${PY} (set PY=...)" >&2; exit 2; }

export ECGR_RUN_TAG="${ECGR_RUN_TAG:-$(date +%y%m%d)_60s}"
export ECGR_IN_CHANNELS="${ECGR_IN_CHANNELS:-3}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export CUDA_VISIBLE_DEVICES="${ECGR_GPU:-0}"

GPU="${ECGR_GPU:-0}"
GPU_FREE_MIB="${ECGR_GPU_FREE_MIB:-18000}"         # wait for this much VRAM before training (60 s x batch 32)
GPU_FREE_INFER_MIB="${ECGR_GPU_FREE_INFER_MIB:-4000}"
GPU_WAIT_MAX="${ECGR_GPU_WAIT_MAX:-21600}"         # ... but no longer than this (s); 0 = never wait

# Largest first: the big sizes are the ones worth a free card early, and a queue that dies
# halfway has then produced the results that matter most.
SIZES=(resumamba_5m resumamba_3m resumamba_1m resumamba_100k)

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

stage_refine() {
    wait_gpu "${GPU_FREE_MIB}"
    echo "[$(date +%H:%M:%S)] refine $1 (temporal head on the frozen base, then no-regression selection on portal-eval)"
    run_stage "${LOGS}/$1_refine.log" \
        "^(Epoch [0-9]|base +:|head +:|refined +:|epoch_|base  |WINNER|refined_best|Weighted F1|Traceback|.*Error)|WINNER|regresses|improves" \
        "${PY}" -m ecgr refine --model "$1" "${@:2}"
}

stage_ec57_refined() {
    local ckpt winner
    ckpt="$("${PY}" -c "from ecgr import config, models; import os; print(os.path.join(config.CHECKPOINT_DIR, models.keras_name('$1'), 'refined', 'refined_best.keras'))")"
    [[ -f "${ckpt}" ]] || { echo "no refined checkpoint for $1: ${ckpt}" >&2; return 1; }
    # The selection can (and, under a strict no-regression rule, often does) pick epoch_00,
    # which IS the base model. Its EC57 numbers already exist under <model>/; re-scoring the
    # same weights under <model>_refined costs ~50 GPU minutes and adds nothing.
    winner="$("${PY}" -c "import json, os; print(json.load(open(os.path.join(os.path.dirname('${ckpt}'), 'selection.json')))['winner']['epoch'])" 2>/dev/null || echo unknown)"
    if [[ "${winner}" == "epoch_00" ]]; then
        echo "[$(date +%H:%M:%S)] $1: selection kept the base (epoch_00) - no separate EC57 to run"
        return 0
    fi
    wait_gpu "${GPU_FREE_INFER_MIB}"
    echo "[$(date +%H:%M:%S)] ec57 $1_refined"
    run_stage "${LOGS}/$1_refined_ec57.log" \
        "(^=====|^EC57 report|^  (Average|Gross|Total)|^EC57 summary|^  db=|Traceback|.*Error)" \
        "${PY}" -m ecgr ec57 --model "$1" --checkpoint "${ckpt}" --tag "$1_refined"
}

# The checkpoint the eval stages score: the bxb-selected epoch when `select` has run, else
# the step-F1 pick (ecgr's own default).
selected_ckpt() {
    "${PY}" -c "from ecgr import config, models; import os; p=os.path.join(config.CHECKPOINT_DIR, models.keras_name('$1'), 'epochs', 'selected_by_bxb.keras'); print(p if os.path.exists(p) else '')"
}

stage_select() {
    wait_gpu "${GPU_FREE_INFER_MIB}"
    echo "[$(date +%H:%M:%S)] select $1 (bxb on portal-eval over every saved epoch)"
    run_stage "${LOGS}/$1_select.log" \
        "(^candidate|^step-F1|^epoch_|WINNER|regresses|selected_by_bxb|Traceback|.*Error)" \
        "${PY}" -m ecgr select --model "$1"
}

stage_stepeval() {
    local ckpt; ckpt="$(selected_ckpt "$1")"
    wait_gpu "${GPU_FREE_INFER_MIB}"
    echo "[$(date +%H:%M:%S)] stepeval $1${ckpt:+ (bxb-selected checkpoint)}"
    run_stage "${LOGS}/$1_stepeval.log" \
        "(Weighted F1|Lead quality|^ *(None|N|V|S) |report ->|checkpoint:|Traceback|.*Error)" \
        "${PY}" -m ecgr stepeval --model "$1" ${ckpt:+--checkpoint "${ckpt}"}
}

stage_ec57() {
    local ckpt; ckpt="$(selected_ckpt "$1")"
    wait_gpu "${GPU_FREE_INFER_MIB}"
    echo "[$(date +%H:%M:%S)] ec57 $1 (physionet native 2 leads + zero fill, portal 3-lead)${ckpt:+ (bxb-selected checkpoint)}"
    run_stage "${LOGS}/$1_ec57.log" \
        "(^=====|^EC57 report|^  (Average|Gross|Total)|^EC57 summary|^  db=|lead quality|geometry|Traceback|.*Error)" \
        "${PY}" -m ecgr ec57 --model "$1" ${ckpt:+--checkpoint "${ckpt}"}
}

stage_regress() {
    echo "[$(date +%H:%M:%S)] regress $1 (against the 10 s baseline in assets/baselines/)"
    set +e
    "${PY}" -m ecgr regress --model "$1" 2>&1 | tee "${LOGS}/$1_regress.log"
    local rc=${PIPESTATUS[0]}
    set -e
    [[ ${rc} -eq 0 ]] || { echo "REGRESSION (rc=${rc}), see ${LOGS}/$1_regress.log" >&2; return ${rc}; }
}

STAGE="${1:?stage: data|sweep|refine|summary|regress|test}"
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
        if [[ -f "${CKPT_ROOT}/${keras}/epochs/selected_by_bxb.keras" ]]; then
            echo "[$(date +%H:%M:%S)] ${m}: bxb selection present, skipping select"
        elif ls "${CKPT_ROOT}/${keras}/epochs/"epoch_*.keras >/dev/null 2>&1; then
            stage_select "$m" || failures+=("${m}:select")
        fi
        stage_stepeval "$m" || failures+=("${m}:stepeval")
        stage_ec57 "$m"     || failures+=("${m}:ec57")
        stage_regress "$m"  || failures+=("${m}:regress")
    done

    "${PY}" -m ecgr compare "${list[@]}" || true
    if [[ ${#failures[@]} -gt 0 ]]; then
        echo "=== ${STAGE} FINISHED WITH FAILURES: ${failures[*]} ==="
        echo "    rerun the same command - finished stages are skipped, only failed ones redo"
        exit 1
    fi
    echo "=== ${STAGE} complete ($(date +%H:%M:%S)) ==="
    ;;

  refine)
    if [[ $# -gt 0 ]]; then list=("$@"); else list=("${SIZES[@]}"); fi
    failures=()
    for m in "${list[@]}"; do
        stage_refine "$m"       || { failures+=("${m}:refine"); continue; }
        stage_ec57_refined "$m" || failures+=("${m}:ec57_refined")
    done
    pairs=(); for m in "${list[@]}"; do pairs+=("$m" "${m}_refined"); done
    "${PY}" -m ecgr compare "${pairs[@]}" || true
    if [[ ${#failures[@]} -gt 0 ]]; then
        echo "=== ${STAGE} FINISHED WITH FAILURES: ${failures[*]} ==="; exit 1
    fi
    echo "=== ${STAGE} complete ($(date +%H:%M:%S)) ==="
    ;;

  summary)
    "${PY}" -m ecgr compare "${@:-${SIZES[@]}}"
    ;;

  regress)
    if [[ $# -gt 0 ]]; then list=("$@"); else list=("${SIZES[@]}"); fi
    failures=()
    for m in "${list[@]}"; do stage_regress "$m" || failures+=("$m"); done
    if [[ ${#failures[@]} -gt 0 ]]; then
        echo "=== regressions in: ${failures[*]} ==="; exit 1
    fi
    echo "=== no regression against the 10 s baselines ==="
    ;;

  *) echo "unknown stage ${STAGE}" >&2; exit 2 ;;
esac
