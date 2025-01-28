from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("amuvarma/3b-zuckreg-convo")

tokenizer.save_pretrained("./zuckreg-llava")