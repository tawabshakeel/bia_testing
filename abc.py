import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("phi1_5",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
    )
tokenizer = AutoTokenizer.from_pretrained("phi1_5",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16)

sys_prompt = "I am OrcaPhi. The following is my internal dialogue as an AI assistant.\n" \
    "Today is September 15, 2023. I have no access to outside tools, news, or current events.\n" \
    "I carefully provide accurate, factual, thoughtful, nuanced answers and am brilliant at reasoning.\n" \
    "I think through my answers step-by-step to be sure I always get the right answer.\n" \
    "I think more clearly if I write out my thought process in a scratchpad manner first; therefore, I always " \
    "explain background context, assumptions, and step-by-step thinking BEFORE trying to answer a question." \
    "Take a deep breath and think calmly about everything presented."
prompt = "Hello! Tell me about what makes you special, as an AI assistant.\n" \
    "Particularly, what programming tasks are you best at?"

prefix = "<|im_start|>"
suffix = "<|im_end|>\n"
sys_format = prefix + "system\n" + sys_prompt + suffix
user_format = prefix + "user\n" + prompt + suffix
assistant_format = prefix + "assistant\n"
input_text = sys_format + user_format + assistant_format

generation_config = GenerationConfig(
    max_length=256,  top_p=0.95,
    do_sample=True, use_cache=True
    )

inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False)
outputs = model.generate(**inputs, generation_config=generation_config)

text = tokenizer.batch_decode(outputs)[0]
print(text)
