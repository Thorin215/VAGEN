import os
from huggingface_hub import HfApi, upload_folder

# 你的 Hugging Face 用户名
HF_USER = "Davidwang215"

# 本地父目录
BASE_DIR = "/raid/users/wc/VAGEN/checkpoints/vagen_7b_full_tool_step3/test-masked_grpo-detectagent"

api = HfApi()

# 提取 checkpoints/ 后面的目录名
checkpoints_root = BASE_DIR.split("checkpoints/")[1].split("/")[0]

for d in os.listdir(BASE_DIR):
    step_dir = os.path.join(BASE_DIR, d)
    target_dir = os.path.join(step_dir, "actor/huggingface")

    if os.path.isdir(target_dir):
        # repo_name = {checkpoints_root}_{global_step_xxx}
        repo_name = f"{checkpoints_root}_{d}"
        repo_id = f"{HF_USER}/{repo_name}"

        print(f"\n>>> 上传 {target_dir} 到 {repo_id}")

        # 创建仓库（如果存在则跳过）
        api.create_repo(repo_id, repo_type="model", exist_ok=True)

        # 上传整个目录
        upload_folder(
            folder_path=target_dir,
            repo_id=repo_id,
            repo_type="model",
        )
