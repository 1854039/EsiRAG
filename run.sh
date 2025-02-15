#!/bin/bash

# 配置参数
# BASE_URL="https://api.deepseek.com/v1"
# API_KEY="sk-60df45081c82468aad5adbb4c0011dae"
BASE_URL="https://aihubmix.com/v1"
API_KEY="sk-aIMN8VDybSdHgvwg439657417a5e4418B8701e444607C690"  # 替换为你的 API key
# BASE_URL="https://aihubmix.com/v1"
# API_KEY="sk-aIMN8VDybSdHgvwg439657417a5e4418B8701e444607C690"  # 替换为你的 API key
# 定义实验参数数组
MODELS=(
    "gemini-2.0-flash"
    #"mistralai/mistral-7b-instruct-v0.3"
    
      # 添加其他模型
)

CORPORA=(
    "law"

    
)

INSERT_MODES=(
    "origin"
)

QUERY_MODES=(
    "local"
)

# 设置最大并行任务数
MAX_PARALLEL_JOBS=6


# 获取开始时间
start_time=$(date +%s)

# 计数器
total_experiments=$((${#MODELS[@]} * ${#CORPORA[@]} * ${#INSERT_MODES[@]} * ${#QUERY_MODES[@]}))
current_experiment=0
running_jobs=0

run_experiment() {
    local model=$1
    local corpus=$2
    local insert_mode=$3
    local query_mode=$4
    local log_file=$5
    
    echo "======================================"
    echo "Starting experiment:"
    echo "Model: $model"
    echo "Corpus: $corpus"
    echo "Insert mode: $insert_mode"
    echo "Query mode: $query_mode"
    echo "Logging to: $log_file"
    echo "======================================"
    
    python run.py \
        --base_url "$BASE_URL" \
        --api_key "$API_KEY" \
        --model "$model" \
        --corpus "$corpus" \
        --insert_mode "$insert_mode" \
        --query_mode "$query_mode" \
        2>&1 | tee "$log_file"
}

# 遍历所有组合
for model in "${MODELS[@]}"; do
    for corpus in "${CORPORA[@]}"; do
        for insert_mode in "${INSERT_MODES[@]}"; do
            for query_mode in "${QUERY_MODES[@]}"; do
                ((current_experiment++))
                
                # 清理文件名中的特殊字符
                safe_model_name=$(echo "$model" | tr '/' '_')
                log_file="logs/${safe_model_name}_${corpus}_${insert_mode}_${query_mode}.log"
                
                # 检查正在运行的任务数
                while [ $(jobs -p | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
                    # 等待任意一个任务完成
                    wait -n
                done
                
                # 在后台运行实验
                run_experiment "$model" "$corpus" "$insert_mode" "$query_mode" "$log_file" &
                
                # 计算和显示进度
                progress=$((current_experiment * 100 / total_experiments))
                echo "Progress: $progress% ($current_experiment/$total_experiments)"
                echo "--------------------------------------"
            done
        done
    done
done

# 等待所有后台任务完成
wait

# 计算总运行时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "======================================"
echo "All experiments completed!"
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Total experiments: $total_experiments"
echo "Logs saved in: ./logs/"
echo "======================================"
