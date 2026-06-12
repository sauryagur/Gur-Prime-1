from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs",  # local checkpoint
    max_seq_length=2048,
    load_in_4bit=True,
)

# Save merged 16-bit first, then quantize to GGUF
model.save_pretrained_gguf(
    "outputs-gguf",
    tokenizer,
    quantization_method="q4_k_m",  # good balance of size/quality
)
