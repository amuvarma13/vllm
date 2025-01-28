mdn = "./zuckreg-llava"
from vllm import  LLM, SamplingParams
import torch

# tokens = tkn(prompt, return_tensors="pt")
# print(tokens)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=mdn)

image_embeds = torch.randn( 1, 3072)
print(image_embeds.shape)
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

print(outputs)