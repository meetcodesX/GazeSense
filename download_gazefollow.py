from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vikhyatk/gazefollow",
    repo_type="dataset",
    local_dir="./gazefollow"
)

print("GazeFollow dataset downloaded successfully!")