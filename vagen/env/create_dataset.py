import os
from vagen.env import REGISTERED_ENV
from vagen.env.detectagent.prompt import system_prompt as detect_system_prompt, format_prompt as DETECT_FORMAT_PROMPT
import numpy as np
import yaml
import argparse
from datasets import Dataset, load_dataset
from typing import List, Dict
import glob
from vagen.env.utils.env_utils import permanent_seed
def _load_image_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def _gather_images(image_dir: str, image_glob: str = "*.{jpg,jpeg,png,bmp}") -> List[str]:
    """Gather image file paths from a directory using glob patterns.
    Supports brace expansion via glob (bash-like) by splitting patterns.
    """
    patterns = []
    if "{" in image_glob and "}" in image_glob:
        # expand manually for common cases
        base, brace = image_glob.split("{")
        suffix, tail = brace.split("}")
        exts = [e.strip() for e in suffix.split(",")]
        for e in exts:
            patterns.append(base + e + tail)
    else:
        patterns = [image_glob]

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(image_dir, pat)))
    files = sorted(list(dict.fromkeys(files)))  # unique & stable order
    return files


def _make_detectagent_samples_from_images(image_paths: List[str], split: str, env_name: str, env_config: dict) -> List[Dict]:
    samples: List[Dict] = []
    sys_text = detect_system_prompt(format="first_prompt")
    user_text = DETECT_FORMAT_PROMPT["first_prompt"]()
    for idx, p in enumerate(image_paths):
        try:
            img_bytes = _load_image_bytes(p)
        except Exception as e:
            print(f"[skip] fail to read {p}: {e}")
            continue
        sample = {
            "data_source": env_name,
            "prompt": [
                {"role": "system", "content": sys_text},
                # The DETECT_FORMAT_PROMPT already describes the required format; prepend <image> token explicitly if desired by your chat template.
                {"role": "user", "content": "<image>\n" + user_text},
            ],
            "images": [{"bytes": img_bytes}],
            "extra_info": {
                "split": split,
                "env_name": env_name,
                "env_config": {**env_config, "image_path": p},
                "index": idx,
                # rollout_manager_service.reset expects a 'seed' per env config
                "seed": idx,
            },
        }
        samples.append(sample)
    return samples


def create_dataset_from_yaml(yaml_file_path: str, force_gen=False,seed=42,train_path='./train.parquet',test_path='./test.parquet'):
    """
    Create dataset from a YAML configuration file.
    
    Args:
        yaml_file_path (str): Path to the YAML configuration file
        force_gen (bool): Whether to force regeneration of existing datasets
        seed (int): Seed for random number generation
        train_path (str): Path to save the training dataset
        test_path (str): Path to save the testing dataset
        
    The YAML file should have the following structure:
    ```
    env1:
        env_name: sokoban  # or frozenlake
        env_config:
            # parameters to override the default env config
        train_size: 100  # number of instances
        test_size:100
    env2:
        env_name: frozenlake
        env_config:
            # parameters to override the default env config
        train_size: 100  # number of instances
        test_size:100
    ```
    
    If the environment config class (e.g., SokobanEnvConfig, FrozenLakeEnvConfig) has a 
    generate_seeds(size) method, it will be used to generate seeds for that environment.
    """
    
    if isinstance(yaml_file_path, str):
        with open(yaml_file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = yaml_file_path
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    if not force_gen and os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Dataset files already exist at {train_path} and {test_path}. Skipping generation.")
        print(f"Use --force-gen to override and regenerate the dataset.")
        return train_path, test_path
    
    
    train_instances = []
    test_instances = []
    
    global_seed = seed
    permanent_seed(global_seed)
    
    
    for key, value in yaml_config.items():
        env_name = value.get('env_name')
        custom_env_config = value.get('env_config', {}) or {}
        train_size, test_size = (value.get('train_size', 100), value.get('test_size', 100))

        # Optional image-based generation configs (support both top-level and mistakenly nested in env_config)
        train_image_dir = value.get('train_image_dir')
        test_image_dir = value.get('test_image_dir')
        image_glob = value.get('image_glob', "*.{jpg,jpeg,png,bmp}")

        # If users mistakenly put these under env_config, extract and remove them
        for k in ['train_image_dir', 'test_image_dir', 'image_glob']:
            if k in custom_env_config:
                if k == 'train_image_dir' and not train_image_dir:
                    train_image_dir = custom_env_config.pop(k)
                elif k == 'test_image_dir' and not test_image_dir:
                    test_image_dir = custom_env_config.pop(k)
                elif k == 'image_glob' and image_glob == "*.{jpg,jpeg,png,bmp}":
                    image_glob = custom_env_config.pop(k)

        # If image dirs provided, build multi-modal samples; otherwise fallback to seed-based placeholders
        if train_image_dir or test_image_dir:
            if train_image_dir:
                train_imgs = _gather_images(train_image_dir, image_glob)
                if train_size:
                    train_imgs = train_imgs[: int(train_size)]
                print(f"Collected {len(train_imgs)} train images from {train_image_dir}")
                train_instances.extend(
                    _make_detectagent_samples_from_images(train_imgs, split="train", env_name=env_name, env_config=custom_env_config)
                )
            if test_image_dir:
                test_imgs = _gather_images(test_image_dir, image_glob)
                if test_size:
                    test_imgs = test_imgs[: int(test_size)]
                print(f"Collected {len(test_imgs)} test images from {test_image_dir}")
                test_instances.extend(
                    _make_detectagent_samples_from_images(test_imgs, split="test", env_name=env_name, env_config=custom_env_config)
                )
            continue

        # Fallback: seed-based placeholder (text-only)
        env_config = REGISTERED_ENV[env_name]["config_cls"](**custom_env_config)
        # seeds_for_env_train = None
        # seeds_for_env_test = None
        if hasattr(env_config, 'generate_seeds'):
            seeds_for_env_train = env_config.generate_seeds(train_size)
            seeds_for_env_test = env_config.generate_seeds(test_size)
            print(f"Using {len(seeds_for_env_train)} trian seeds generated by {env_name} config's generate_seeds method")
            print(f"Using {len(seeds_for_env_test)} test seeds generated by {env_name} config's generate_seeds method")
        else:
            seeds_for_env_train = np.random.randint(0, 2**31 - 1, size=train_size).tolist()
            seeds_for_env_test = np.random.randint(0, 2**31 - 1, size=test_size).tolist()

        for s in seeds_for_env_train:
            env_settings = { 'env_name': env_name, 'env_config': custom_env_config, 'seed': s }
            instance = {
                "data_source": env_name,
                "prompt": [{"role": "user", "content": ''}],
                "extra_info": {"split": "train", **env_settings}
            }
            train_instances.append(instance)
        for s in seeds_for_env_test:
            env_settings = { 'env_name': env_name, 'env_config': custom_env_config, 'seed': s }
            instance = {
                "data_source": env_name,
                "prompt": [{"role": "user", "content": ''}],
                "extra_info": {"split": "test", **env_settings}
            }
            test_instances.append(instance)
            
    
    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn
        
    # Create datasets
    if train_instances:
        train_dataset = Dataset.from_list(train_instances)
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        train_dataset.to_parquet(train_path)
        print(f"Train dataset with {len(train_instances)} instances saved to {train_path}")
    
    if test_instances:
        test_dataset = Dataset.from_list(test_instances)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        test_dataset.to_parquet(test_path)
        print(f"Test dataset with {len(test_instances)} instances saved to {test_path}")
    
    if not train_instances and not test_instances:
        print("No instances were generated. Check your YAML configuration.")
    
    return train_path, test_path
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--force_gen", action="store_true", help="Force regenerate dataset even if exists")
    parser.add_argument("--train_path", type=str, default="./train.parquet", help="Path to save the training dataset")
    parser.add_argument("--test_path", type=str, default="./test.parquet", help="Path to save the testing dataset")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    args = parser.parse_args()
    print(args)
    train_path, test_path = create_dataset_from_yaml(args.yaml_path, args.force_gen, args.seed, args.train_path, args.test_path)
    
    # Optionally load the dataset and print examples
    train_dataset = load_dataset('parquet', data_files={"train": train_path}, split="train")
    test_dataset = load_dataset('parquet', data_files={"test": test_path}, split="test")
    for i in range(2):
        print(train_dataset[i])
        env_name = train_dataset[i]["extra_info"]["env_name"]
        env_config_cls = REGISTERED_ENV[env_name]["config_cls"]
        env_config= env_config_cls(**train_dataset[i]["extra_info"]["env_config"])
        print(env_config.config_id())
    for i in range(2):
        print(train_dataset[i])
        env_name = test_dataset[i]["extra_info"]["env_name"]
        env_config_cls = REGISTERED_ENV[env_name]["config_cls"]
        env_config= env_config_cls(**test_dataset[i]["extra_info"]["env_config"])
        print(env_config.config_id())