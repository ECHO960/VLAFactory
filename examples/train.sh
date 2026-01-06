set -e
set -x

############################
# paths (user configurable)
############################
LLAMA_FACTORY_DIR=../third_party/LLaMA-Factory
LLAMA_FACTORY_TRAIN=${LLAMA_FACTORY_DIR}/src/train.py

PYTORCH_KERNEL_CACHE_PATH=/opt/tiger/caches
save_checkpoint_path=/opt/tiger/local_checkpoints
MODEL_HDFS_PATH=hdfs://harunavaali/home/byte_search_general_ranking_us/chunhui.liu/models/Qwen3-VL-2B-Instruct
JOB_CONFIG=./qwen3_vl_8b_sft.yaml

save_hdfs_dir=hdfs://harunava/home/byte_search_general_ranking_us/chunhui.liu/checkpoints
checkpoint_subdir=KD-Test-Qwen3-VL-2B-Instruct
############################

pwd
script_dir="$(dirname "$0")"
echo $script_dir
cd $script_dir
pwd

pip3 install -e ${LLAMA_FACTORY_DIR}
pip3 uninstall gradio -y
pip3 install deepspeed joblib
pip3 install transformers==4.57.0

export PYTORCH_KERNEL_CACHE_PATH=${PYTORCH_KERNEL_CACHE_PATH}
mkdir -p ${PYTORCH_KERNEL_CACHE_PATH}

mkdir -p ${save_checkpoint_path}

# download checkpoints
hadoop fs -get ${MODEL_HDFS_PATH} ${save_checkpoint_path}
ls -lh ${save_checkpoint_path}

# tmp
cp ${JOB_CONFIG} ${save_checkpoint_path}/

torchrun --nproc_per_node ${ARNOLD_WORKER_GPU} \
        --nnodes ${ARNOLD_WORKER_NUM} \
        --node_rank ${ARNOLD_ID} \
        --master_addr ${METIS_WORKER_0_HOST} \
        --master_port ${METIS_WORKER_0_PORT} \
    ${LLAMA_FACTORY_DIR}/src/train.py ${JOB_CONFIG}

hadoop fs -mkdir -p ${save_hdfs_dir}/${checkpoint_subdir}
hadoop fs -put ${save_checkpoint_path}/checkpoint-* ${save_hdfs_dir}/${checkpoint_subdir}

sleep 28800
