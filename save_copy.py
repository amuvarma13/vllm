# mdn = "amuvarma/3b-zuckreg-convo"
# from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration
# # model = AutoModelForCausalLM.from_pretrained(mdn)
# model = LlavaForConditionalGeneration.from_pretrained(mdn)
# print(model)
# # model.save_pretrained("./zuckreg")

from transformers import LlavaForConditionalGeneration, AutoModel

checkpoint_name = "amuvarma/3b-zuckreg-convo"
config_path = "llavaconfig.json"

model = LlavaForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=checkpoint_name,
    config=config_path,
    ignore_mismatched_sizes=True
)

print(model)

llm_model = AutoModel.from_pretrained(checkpoint_name)
print(llm_model)