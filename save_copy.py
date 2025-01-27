mdn = "amuvarma/3b-zuckreg-convo"
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(mdn)
print(model)