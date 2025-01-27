# mdn = "amuvarma/3b-zuckreg-convo"
# from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration
# # model = AutoModelForCausalLM.from_pretrained(mdn)
# model = LlavaForConditionalGeneration.from_pretrained(mdn)
# print(model)
# # model.save_pretrained("./zuckreg")


checkpoint_name = "amuvarma/3b-zuckreg-convo"
config_path = "llavaconf.json"

from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig, AutoModel

llm_model = AutoModel.from_pretrained(checkpoint_name)
print(llm_model)

# Initializing a CLIP-vision config
vision_config = CLIPVisionConfig()

# Initializing a Llama config
text_config = llm_model.config

# Initializing a Llava llava-1.5-7b style configuration
configuration = LlavaConfig(vision_config, text_config)

# Initializing a model from the llava-1.5-7b style configuration
model = LlavaForConditionalGeneration(configuration)


print(model)
print(llm_model)