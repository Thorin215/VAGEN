#!/bin/bash

# 配置参数
BASE_DIR="/raid/users/wc/VAGEN/checkpoints/vagen_7b_full_tool_step_data/test-masked_grpo-detectagent"
MERGE_SCRIPT="/raid/users/wc/verl/scripts/model_merger.py"
DRY_RUN=false  # 设置为true时只显示操作不会实际执行

# 开始处理
echo "===== 开始批量模型合并与清理 ====="
echo "基础目录: ${BASE_DIR}"
echo "合并脚本: ${MERGE_SCRIPT}"
echo "模式: ${DRY_RUN}"

# 遍历所有global_step目录
for step_dir in "${BASE_DIR}"/global_step_*/; do
    STEP_NAME=$(basename "${step_dir}")
    ACTOR_DIR="${step_dir}actor"
    
    echo ""
    echo ">>>> 正在处理 ${STEP_NAME} <<<<"
    
    # 检查actor目录是否存在
    if [ ! -d "${ACTOR_DIR}" ]; then
        echo "跳过: ${ACTOR_DIR} 不存在"
        continue
    fi
    
    # 步骤1: 执行模型合并
    echo "▶ 执行模型合并..."
    if [ "$DRY_RUN" = false ]; then
        python3 "${MERGE_SCRIPT}" --local_dir "${ACTOR_DIR}"
    else
        echo "[DRY RUN] 将执行: python3 ${MERGE_SCRIPT} --local_dir ${ACTOR_DIR}"
    fi
    
    # 检查合并是否成功
    if [ ! -d "${ACTOR_DIR}/huggingface" ]; then
        echo "⚠️ 警告: 合并可能失败，${ACTOR_DIR}/huggingface 不存在"
        continue
    fi
    
    # 步骤2: 清理文件（保留所有文件夹）
    echo "▶ 开始清理文件（保留所有文件夹）..."
    if [ "$DRY_RUN" = false ]; then
        cd "${ACTOR_DIR}" || exit 1
        # 安全删除：先打印将要删除的内容
        echo "以下文件将被删除（保留所有文件夹）:"
        find . -maxdepth 1 -type f -print
        
        # 实际执行删除
        find . -maxdepth 1 -type f -exec rm -f {} +
        echo "文件清理完成"
    else
        echo "[DRY RUN] 将删除以下文件:"
        find "${ACTOR_DIR}" -maxdepth 1 -type f -print
    fi
    
    echo "√ 完成处理 ${STEP_NAME}"
done

echo ""
echo "===== 所有任务处理完成 ====="