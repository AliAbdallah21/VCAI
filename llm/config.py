# llm/config.py
"""LLM Configuration"""

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# BitsAndBytes 4-bit quantization settings
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True
}

# Generation settings
GENERATION_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}
