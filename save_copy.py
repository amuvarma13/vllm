# mdn = "amuvarma/3b-zuckreg-convo"
# from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration
# # model = AutoModelForCausalLM.from_pretrained(mdn)
# model = LlavaForConditionalGeneration.from_pretrained(mdn)
# print(model)
# # model.save_pretrained("./zuckreg")

from transformers import LlavaForCausalLM

checkpoint_name = "amuvarma/3b-zuckreg-convo"

model = LlavaForCausalLM.from_pretrained(checkpoint_name)

print(model)
# model = LlavaForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=checkpoint_name,
#     config=config_path,
#     ignore_mismatched_sizes=True
# )

