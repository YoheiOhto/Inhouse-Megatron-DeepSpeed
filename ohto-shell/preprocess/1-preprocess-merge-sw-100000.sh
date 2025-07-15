#!/bin/sh
#PBS -q regular-c
#PBS -l select=1
#PBS -W group_list=gd43
#PBS -o preprocess_merged.out
#PBS -e preprocess_merged.err

module purge
module load cmake
module load gcc

source /work/gg17/a97006/.c_bashrc
cd ~/env/env-c
source ./250/bin/activate

wandb login 65afaa936940cf3a198fba3da2d51b71b797b77e
set -e

# --- 基本設定 ---
BASE_DIR="/work/gg17/a97006/250519_modern_bert_0"
SCRIPT_PATH="${BASE_DIR}/Inhouse-Megatron-DeepSpeed/tools/preprocess_data_sw.py"
VOCAB_FILE="${BASE_DIR}/tokenizer/vocab_100000.txt"
TOKENIZER_TYPE="BertWordPieceCase"
WORKERS=12
WANDB_PROJECT="med_preprocess"
DATE_TAG="250708"

# --- ステップ1：データソースの結合 ---
echo "=================================================="
echo "Step 1: Merging source JSONL files..."
echo "=================================================="

# 入力ファイルパス
PUBMED_JSON="${BASE_DIR}/json/pubmed.jsonl"
NIH_JSON="${BASE_DIR}/json/nih_books.jsonl"
FDA_JSON="${BASE_DIR}/json/fda_label.jsonl"
PMC_JSON="${BASE_DIR}/json/pmc.jsonl"

# 出力ファイルパス
MERGED_JSON_DIR="${BASE_DIR}/json"
mkdir -p "${MERGED_JSON_DIR}"
MERGED_JSON_FILE="${MERGED_JSON_DIR}/4_merged_data.jsonl"

# catコマンドでファイルを結合（PMCを含める場合は ${PMC_JSON} を追加）
cat "${PUBMED_JSON}" "${NIH_JSON}" "${FDA_JSON}" "${PMC_JSON}" | shuf > "${MERGED_JSON_FILE}"

echo "Finished merging files into: ${MERGED_JSON_FILE}"
echo ""


# --- ステップ2：結合したファイルの前処理 ---
echo "=================================================="
echo "Step 2: Preprocessing the merged file..."
echo "=================================================="

# 前処理後の出力先
OUTPUT_PREFIX_DIR="${BASE_DIR}/preprocessed/4_merged/all_merged_100000-1024"
mkdir -p "${OUTPUT_PREFIX_DIR}"
OUTPUT_PREFIX="${OUTPUT_PREFIX_DIR}/4_merged"
WANDB_NAME="${DATE_TAG}-miyabi-4_merged-100000-1024"


python "${SCRIPT_PATH}" \
    --input "${MERGED_JSON_FILE}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --vocab-file "${VOCAB_FILE}" \
    --tokenizer-type "${TOKENIZER_TYPE}" \
    --workers "${WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --seq-length 1021 \
    --sliding-window-stride 512 \

echo ""
echo "All datasets have been processed into a single dataset."