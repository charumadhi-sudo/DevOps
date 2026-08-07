import os
import sys
from huggingface_hub import HfApi, create_repo

def main():
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID")
    
    if not token:
        print("[register] ERROR: HF_TOKEN environment variable not set.")
        sys.exit(1)
        
    if not repo_id:
        print("[register] ERROR: HF_REPO_ID environment variable not set.")
        sys.exit(1)
        
    print(f"[register] Target Hugging Face Repository: {repo_id}")
    
    api = HfApi()
    
    # Create repo if not exist
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"[register] Repository {repo_id} exists or was created successfully.")
    except Exception as e:
        print(f"[register] Warning during repository check/creation: {e}")
        
    # Upload folder
    try:
        print(f"[register] Uploading 'model' directory...")
        api.upload_folder(
            folder_path="model",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("[register] Successfully uploaded model artifacts.")
        
        # Upload metrics
        print(f"[register] Uploading 'metrics.json'...")
        api.upload_file(
            path_or_fileobj="metrics.json",
            path_in_repo="metrics.json",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("[register] Successfully uploaded metrics.json.")
        
    except Exception as e:
        print(f"[register] ERROR during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
