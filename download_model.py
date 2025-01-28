from huggingface_hub import snapshot_download

mdn = "amuvarma/3b-zuckreg-convo"


model_path = snapshot_download(
    repo_id=mdn,
    allow_patterns=[
        "config.json",
        "*.safetensors",
        "model.safetensors.index.json",
    ],
    ignore_patterns=[
        "optimizer.pt",
        "pytorch_model.bin",
        "training_args.bin",
        "scheduler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.*"
    ]
)

from transformers import AutoModel 
model = AutoModel.from_pretrained(model_path)
model.save_pretrained("zuckreg") 