#!/bin/sh
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=6:00:00
#PBS -W group_list=gd43
#PBS -o ccc.out
#PBS -e ccc.err

# --- 0. 環境設定 ---
echo "--- 環境を設定しています... ---"
module purge
module load cmake
module load gcc
module load cuda/12.6
module load cudnn/9.5.1.17
module load ompi-cuda/4.1.6-12.6

source /work/gg17/a97006/.g_bashrc
cd ~/env/llm-pyenv-4
source ./250/bin/activate

# pip uninstall -y triton

# git clone https://github.com/triton-lang/triton.git
# cd triton
# git checkout release/3.2.x
# pip install ninja cmake wheel pybind11
# pip install -e python

export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.6"
export LD_LIBRARY_PATH=/work/opt/local/aarch64/cores/jupyterlab/4.3.5/python/3.12.8/lib:\
$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset OMPI_MCA_mca_base_env_list

export PYTHONPATH="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed"

# --- 分散実行用の設定 (ループの前に一度だけ実行) ---
UNIQUE_NODES_FILE=$(mktemp)
if [ -z "$PBS_NODEFILE" ]; then
    echo "Error: PBS_NODEFILE is not set. Running on localhost."
    echo "$(hostname)" > $UNIQUE_NODES_FILE
    num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    num_node=1
else
    sort -u $PBS_NODEFILE > $UNIQUE_NODES_FILE
fi

if [ -s $UNIQUE_NODES_FILE ]; then
    export MASTER_ADDR=$(head -n 1 $UNIQUE_NODES_FILE)
else
    export MASTER_ADDR=$(hostname) # Fallback for local/single node
fi
export MASTER_PORT=29500 # Choose a free port


# --- ループ処理 ---
# ### 変更点 1: CHECKPOINT_DIR を新しいパスに更新 ###
CHECKPOINT_DIR="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/a97006/project/bert_with_pile/checkpoint/250906_full_100000-0.149B-iters-2M-lr-8e-4-min-1e-5-wmup-20700-dcy-2M-sty-constant-gbs-1280-mbs-20-gpu-64-zero-0-mp-1-pp-1-nopp"
vocab_path="/work/gg17/a97006/250519_modern_bert_0/tokenizer/vocab_100000.txt"

# ### 変更点 2: 1から10までループ ###
for i in $(seq 1 3)
do
    # ### 変更点 3: ステップ数を20700の倍数で計算 ###
    STEP=$((i * 20700))
    TAG="global_step${STEP}"

    echo ""
    echo "#####################################################"
    echo "### Processing checkpoint for ${TAG}..."
    echo "#####################################################"

    # ステップごとにユニークな出力パスを設定
    OUTPUT_PATH_LOOP="/work/gg17/a97006/hf_models/med-bert-step${STEP}-fp32"
    SAVE_PATH_LOOP="/work/gg17/a97006/hf_models/med-bert-step${STEP}-hf"

    # --- 1. zero to fp32 変換 ---
    echo "--- Running zero_to_fp32.py for ${TAG} ---"
    echo "Outputting to: ${OUTPUT_PATH_LOOP}"
    python3 /work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/ohto-shell/model_convert/zero_to_fp32.py \
        $CHECKPOINT_DIR \
        $OUTPUT_PATH_LOOP \
        --tag ${TAG}
    
    # エラーチェック
    if [ $? -ne 0 ]; then
        echo "Error during zero_to_fp32.py for ${TAG}. Aborting this step."
        continue
    fi

    # --- 2. Megatron to Hugging Face 変換 ---
    # ### 推奨される変更: リポジトリ名をステップごとにユニークに ###
    REPO_NAME="YoheiOhto/250908_1200_${i}"
    echo "--- Running modern_bert_converter_bin.py for ${TAG} ---"
    echo "Saving final model to: ${SAVE_PATH_LOOP}"
    echo "Uploading to Hugging Face repo: ${REPO_NAME}"
    python /work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/tools/convert_checkpoint/modern_bert_converter_bin.py \
        --load ${OUTPUT_PATH_LOOP}/pytorch_model.bin \
        --save ${SAVE_PATH_LOOP} \
        --num-layers 22 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --micro-batch-size 1 \
        --global-batch-size 1 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --save-interval 1000 \
        --vocab-file ${vocab_path} \
        --repo_id "${REPO_NAME}" \
        --tokenizer-type BertWordPieceCase \
        --target_vocab_size 100096
    
    # エラーチェック
    if [ $? -ne 0 ]; then
        echo "Error during modern_bert_converter_bin.py for ${TAG}. Aborting this step."
        continue
    fi

done

echo ""
echo "##############################################"
echo "### 全ての変換プロセスが正常に完了しました！ ###"
echo "##############################################"


# --- 0. 環境設定 ---
echo "--- 環境を設定しています... ---"
module purge
module load cmake
module load gcc
module load cuda/12.6
module load cudnn/9.5.1.17
module load ompi-cuda/4.1.6-12.6

source /work/gg17/a97006/.g_bashrc
cd ~/env/llm-pyenv-4
source ./250/bin/activate

export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.6"
export LD_LIBRARY_PATH=/work/opt/local/aarch64/cores/jupyterlab/4.3.5/python/3.12.8/lib:\
$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset OMPI_MCA_mca_base_env_list

export PYTHONPATH="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed"
export TORCH_CUDA_ARCH_LIST="9.0"

# --- 分散実行用の設定 (ループの前に一度だけ実行) ---
UNIQUE_NODES_FILE=$(mktemp)
if [ -z "$PBS_NODEFILE" ]; then
    echo "Error: PBS_NODEFILE is not set. Running on localhost."
    echo "$(hostname)" > $UNIQUE_NODES_FILE
    num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    num_node=1
else
    sort -u $PBS_NODEFILE > $UNIQUE_NODES_FILE
fi

if [ -s $UNIQUE_NODES_FILE ]; then
    export MASTER_ADDR=$(head -n 1 $UNIQUE_NODES_FILE)
else
    export MASTER_ADDR=$(hostname) # Fallback for local/single node
fi
export MASTER_PORT=29500 # Choose a free port


# --- ループ処理 ---
# ### 変更点 1: CHECKPOINT_DIR を新しいパスに更新 ###
CHECKPOINT_DIR="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/a97006/project/bert_with_pile/checkpoint/250906_full_100000-0.149B-iters-2M-lr-8e-4-min-1e-5-wmup-20700-dcy-2M-sty-constant-gbs-5120-mbs-20-gpu-64-zero-0-mp-1-pp-1-nopp"
vocab_path="/work/gg17/a97006/250519_modern_bert_0/tokenizer/vocab_100000.txt"

# ### 変更点 2: 1から10までループ ###
for i in $(seq 1 3)
do
    # ### 変更点 3: ステップ数を20700の倍数で計算 ###
    STEP=$((i * 20700))
    TAG="global_step${STEP}"

    echo ""
    echo "#####################################################"
    echo "### Processing checkpoint for ${TAG}..."
    echo "#####################################################"

    # ステップごとにユニークな出力パスを設定
    OUTPUT_PATH_LOOP="/work/gg17/a97006/hf_models/med-bert-step${STEP}-fp32"
    SAVE_PATH_LOOP="/work/gg17/a97006/hf_models/med-bert-step${STEP}-hf"

    # --- 1. zero to fp32 変換 ---
    echo "--- Running zero_to_fp32.py for ${TAG} ---"
    echo "Outputting to: ${OUTPUT_PATH_LOOP}"
    python3 /work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/ohto-shell/model_convert/zero_to_fp32.py \
        $CHECKPOINT_DIR \
        $OUTPUT_PATH_LOOP \
        --tag ${TAG}
    
    # エラーチェック
    if [ $? -ne 0 ]; then
        echo "Error during zero_to_fp32.py for ${TAG}. Aborting this step."
        continue
    fi

    # --- 2. Megatron to Hugging Face 変換 ---
    # ### 推奨される変更: リポジトリ名をステップごとにユニークに ###
    REPO_NAME="YoheiOhto/250908_5000_${i}"
    echo "--- Running modern_bert_converter_bin.py for ${TAG} ---"
    echo "Saving final model to: ${SAVE_PATH_LOOP}"
    echo "Uploading to Hugging Face repo: ${REPO_NAME}"
    python /work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/tools/convert_checkpoint/modern_bert_converter_bin.py \
        --load ${OUTPUT_PATH_LOOP}/pytorch_model.bin \
        --save ${SAVE_PATH_LOOP} \
        --num-layers 22 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --micro-batch-size 1 \
        --global-batch-size 1 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --save-interval 1000 \
        --vocab-file ${vocab_path} \
        --repo_id "${REPO_NAME}" \
        --tokenizer-type BertWordPieceCase \
        --target_vocab_size 100096
    
    # エラーチェック
    if [ $? -ne 0 ]; then
        echo "Error during modern_bert_converter_bin.py for ${TAG}. Aborting this step."
        continue
    fi

done

echo ""
echo "##############################################"
echo "### 全ての変換プロセスが正常に完了しました！ ###"
echo "##############################################"