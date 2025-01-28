from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("amuvarma/3b-zuckreg-convo")


print(len(tokenizer))

tokenizer.save_pretrained("./zuckreg-llava")