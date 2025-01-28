mdn = "./zuckreg"
from vllm import ModelRegistry, LLM, SamplingParams
import inspect
from transformers import AutoTokenizer
tkn = AutoTokenizer.from_pretrained(mdn)
import torch

# tokens = tkn(prompt, return_tensors="pt")
# print(tokens)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=mdn)

image_embeds = torch.randn( 1, 3072)
print(image_embeds.shape)

outputs = llm.generate({
    "prompt_embeds": image_embeds,
})

print(outputs)