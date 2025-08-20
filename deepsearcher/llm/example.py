from vllm import LLM, SamplingParams

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
llm = LLM(model="Qwen/Qwen3-8B",gpu_memory_utilization=0.7, tensor_parallel_size=8)

# Prepare the input to the model
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]

# Generate outputs
outputs = llm.chat(
    [messages], 
    sampling_params,
    chat_template_kwargs={"enable_thinking": True},  # Set to False to strictly disable thinking
)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")