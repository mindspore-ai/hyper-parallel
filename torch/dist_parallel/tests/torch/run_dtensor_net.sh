#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=1
export HCCL_TOPO_DETECT_MODE=0 
#export HCCL_IF_BASE_PORT=64000
#export OMP_NUM_THREADS=1
torchrun --nproc-per-node=8 --master-addr=localhost --master-port=29409 test_dtensor_net.py
