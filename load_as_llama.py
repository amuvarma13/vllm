mdn = "./zuckreg-llava"
from vllm import ModelRegistry, LLM, SamplingParams
import inspect

prompt = "Hello my name is"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=mdn)

outputs = llm.generate(prompt, sampling_params)

