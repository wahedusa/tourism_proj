import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Initialize API client
# Ensure HF_TOKEN is set as an environment variable or passed securely
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the target Hugging Face Space ID and type
# IMPORTANT: Replace 'YOUR_HF_USERNAME/YOUR_SPACE_NAME' with your actual Space ID
hf_space_id = "wahedali025/tourism-project"
hf_repo_type = "space"
local_dir = "/content/tourism_project/deployment"

print(f"Attempting to push deployment files from '{local_dir}' to Hugging Face Space '{hf_space_id}'...")

try:
    # Check if the space exists
    api.repo_info(repo_id=hf_space_id, repo_type=hf_repo_type)
    print(f"Space '{hf_space_id}' already exists. Updating files...")
except RepositoryNotFoundError:
    # If the space does not exist, create it, specifying space_sdk
    print(f"Space '{hf_space_id}' not found. Creating new space...")
    create_repo(repo_id=hf_space_id, repo_type=hf_repo_type, private=False, space_sdk='streamlit') # Added space_sdk
    print(f"Space '{hf_space_id}' created.")
except Exception as e:
    print(f"An error occurred while checking or creating the space: {e}")
    exit()

# Upload the entire folder
try:
    api.upload_folder(
        folder_path=local_dir,
        repo_id=hf_space_id,
        repo_type=hf_repo_type,
        commit_message="Update Streamlit app and dependencies"
    )
    print(f"Successfully pushed deployment files to Hugging Face Space '{hf_space_id}'.")
except Exception as e:
    print(f"An error occurred during file upload: {e}")
