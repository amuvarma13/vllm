mdn = "./zuckreg-llava"
from vllm import ModelRegistry, LLM, SamplingParams
import inspect
from transformers import AutoTokenizer
tkn = AutoTokenizer.from_pretrained(mdn)
import torch

prompt = "<image>"
tokens = tkn(prompt, return_tensors="pt")
print(tokens)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=mdn)

image_embeds = torch.randn(1, 3072)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image_embeds},
})

print(outputs)