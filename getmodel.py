from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir=r"E:\airLLM-proj\models\TinyLlama-1.1B-Chat",
    max_workers = 8
)