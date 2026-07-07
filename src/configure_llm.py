from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM


model_dict = {"llama3_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct", 
          "mistral": "mistralai/Mistral-7B-Instruct-v0.3", 
          "gemma4_12b": "google/gemma-4-12B-it",
          "qwen3_8b": "Qwen/Qwen3-8B", 
          "phi_4":"microsoft/phi-4", 
          "qwen3_14b":"Qwen/Qwen3-14B"}

generation_config_dict = {
    "gemma4_12b": {"thinking": {'max_new_tokens':24000, "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64}, "non_thinking": {'max_new_tokens':24000, "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64}},
    "qwen3_8b": {"thinking": {'max_new_tokens':24000, "do_sample": True}, "non_thinking": {'max_new_tokens':24000, "do_sample": True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0}}, 
    "qwen3_14b": {"thinking": {'max_new_tokens':24000, "do_sample": True}, "non_thinking": {'max_new_tokens':24000, "do_sample": True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0}},
}

processor_dict = {
    "gemma4_12b": AutoProcessor.from_pretrained(model_dict["gemma4_12b"]),
    "qwen3_8b": AutoTokenizer.from_pretrained(model_dict["qwen3_8b"]),
    "qwen3_14b": AutoTokenizer.from_pretrained(model_dict["qwen3_14b"]),
}

def prompt_model(model_name, input, num_output, thinking=True):
    if thinking:
        key_think = "thinking"
    else:
        key_think = "non_thinking"
    generation_config = generation_config_dict[model_name][key_think]
    generation_config["num_return_sequences"] = num_output
    model = AutoModelForCausalLM.from_pretrained(model_dict[model_name],dtype="auto",device_map="auto")
    processor = processor_dict[model_name]
    chat = processor.apply_chat_template(input, tokenize=True,return_tensors="pt",return_dict=True, add_generation_prompt=True, enable_thinking=thinking).to(model.device)
    input_len = chat["input_ids"].shape[-1]
    output = model.generate(**chat, **generation_config)
    parsed =[decode_model_output(model_name, o, processor, input_len) for o in output]
    if len(parsed) == 1 :
        return parsed[-1]
    else:
        return parsed

def decode_model_output(model_name, output, processor, input_len):
    if "qwen" in model_name:
        output = output[input_len:].tolist()
        try:
            # rindex finding 151668 (</think>)
            index = len(output) - output[::-1].index(151668)
        except ValueError:
            index = 0

        parsed = {"role": "assistant", "thinking":processor.decode(output[:index], skip_special_tokens=True).strip("\n"), "content":processor.decode(output[index:], skip_special_tokens=True).strip("\n")}

    if "gemma" in model_name:
        response = processor.decode(output[input_len:], skip_special_tokens=False)
        parsed = processor.parse_response(response)
    
    return parsed

