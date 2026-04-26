import os
from huggingface_hub import hf_hub_download

# Default Hugging Face repository ID
DEFAULT_REPO_ID = "saishhh/brain-stroke-model"
# Directory to cache the downloaded models
CACHE_DIR = "model_cache"

FILES_TO_DOWNLOAD = [
    "best_unet_model.h5",
    "my_stroke_ensemble_lightgbm.pkl",
    "my_stroke_ensemble_catboost.pkl",
    "my_stroke_ensemble_adaboost.pkl",
    "my_stroke_ensemble_decision_tree_meta.pkl",
    "my_stroke_ensemble_scaler.pkl"
]

def download_models(repo_id=DEFAULT_REPO_ID, cache_dir=CACHE_DIR):
    """
    Downloads required model files from Hugging Face and caches them locally.
    Returns a dictionary mapping filenames to their absolute cached paths.
    """
    print(f"Checking models from Hugging Face repository: {repo_id}")
    os.makedirs(cache_dir, exist_ok=True)
    
    downloaded_paths = {}
    for filename in FILES_TO_DOWNLOAD:
        print(f"Ensuring {filename} is available...")
        try:
            # hf_hub_download automatically handles caching. It only downloads 
            # if the file isn't already cached or if there's a new version.
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir
            )
            downloaded_paths[filename] = path
            print(f"  -> Ready: {path}")
        except Exception as e:
            print(f"  -> Error downloading {filename}: {e}")
            raise
            
    print("All required models are successfully downloaded and cached!")
    return downloaded_paths

if __name__ == "__main__":
    print("=" * 70)
    print("Downloading Brain Stroke Detection Models")
    print("=" * 70)
    download_models()
