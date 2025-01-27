mdn = "amuvarma/3b-zuckreg-convo"
from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained(mdn)
print(model)
# model.save_pretrained("./zuckreg")