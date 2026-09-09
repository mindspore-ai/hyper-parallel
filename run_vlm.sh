set -o pipefail

OUTPUT_DIR='/cache/hyper-dev/hyper-parallel/output'
PLOG_DIR="${OUTPUT_DIR}/plog"
mkdir -p "${OUTPUT_DIR}" "${PLOG_DIR}"

export ASCEND_PROCESS_LOG_PATH="${PLOG_DIR}"

export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} #￥-8,9,10,11,12,13,14,15


torchrun --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29595 \
    --module examples.training_demo.train_vlm examples/training_demo/train_vlm.yaml \
    2>&1 | tee "${OUTPUT_DIR}/run_vlm.log"
