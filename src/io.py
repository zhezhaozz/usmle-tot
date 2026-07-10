import json
from tqdm import tqdm

from src.prompts import io_prompt
from src.configure_llm import *
from src.input_utils import *
from src.output_utils import parse_json_output
from src.evaluate import evaluate


def run_io(model_name, 
           data_path,
           thinking=False, 
           n_return=5):

    dataset = load_dataset(data_path)
    
    results = []
    for case in tqdm(dataset):
        prompt = construct_prompt(io_prompt, case)
        output = prompt_model(model_name,prompt,n_return,thinking)
        result = parse_json_output(output["content"])
        results.append(result)
    
    accuracy, failure_rate = evaluate(dataset, results)
    return accuracy, failure_rate



