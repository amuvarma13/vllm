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


from huggingface_hub import hf_hub_download
import torch

local_path = hf_hub_download(repo_id="amuvarma/3b-zuckreg-convo", filename="pytorch_model.bin")
state_dict = torch.load(local_path, map_location="cpu")
print(state_dict.keys())
