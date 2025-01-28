from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("amuvarma/3b-zuckreg-convo")

tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
print(len(tokenizer))

tokenizer.save_pretrained("./zuckreg-llava")