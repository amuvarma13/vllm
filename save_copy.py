# mdn = "amuvarma/3b-zuckreg-convo"
# from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration
# # model = AutoModelForCausalLM.from_pretrained(mdn)
# model = LlavaForConditionalGeneration.from_pretrained(mdn)
# print(model)
# # model.save_pretrained("./zuckreg")

# from transformers import AutoModel

# checkpoint_name = "amuvarma/3b-zuckreg-convo"
# config_path = "./llavaconf.json"

# model = AutoModel.from_pretrained(
#     pretrained_model_name_or_path=checkpoint_name,
#     config=config_path,
#     ignore_mismatched_sizes=True
# )

# print(model)


import os
from safetensors.torch import safe_open, save_file

def rename_key(old_key: str) -> str:
    """
    Example rename logic:
    - If your old checkpoint stored keys like `model.layers.X...`
    - and LLaVA expects `model.language_model.model.layers.X...`,
    - you do a string replace:
    """
    if old_key.startswith("model.layers"):
        new_key = old_key.replace("model.layers", "model.language_model.model.layers")
    elif old_key == "model.embed_tokens.weight":
        new_key = "model.language_model.model.embed_tokens.weight"
    else:
        # fallback prefix
        new_key = "model.language_model." + old_key
    return new_key

shard_dir = "zuckreg"  # path to folder with multiple *.safetensors files
shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".safetensors"))

for shard_file in shard_files:
    shard_path = os.path.join(shard_dir, shard_file)
    print(f"Processing shard: {shard_path}")

    new_state_dict = {}

    # safe_open lets us iterate over tensor keys without loading all into memory at once
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for old_key in f.keys():
            tensor = f.get_tensor(old_key)
            # rename
            new_key = rename_key(old_key)
            new_state_dict[new_key] = tensor

    # Save to a new shard file (or overwrite if you prefer)
    renamed_path = os.path.join(shard_dir, f"renamed-{shard_file}")
    save_file(new_state_dict, renamed_path)

    print(f"Renamed shard saved to: {renamed_path}")
