import argparse
import os
os.environ['HF_HOME'] = '/nfs/turbo/umms-vgvinodv/users/zzhaozhe/cache'
import random
import torch
import numpy as np
from scipy import stats

from src.io import run_io
from src.tot import run_tot
from src.input_utils import DATA_FOLDER

def mean_ci_95(numbers):
    print(numbers)
    data = np.array(numbers, dtype=float)
    print(data)
    n = len(data)

    mean = np.mean(data)
    if mean == 0.0:
        return "0 (0.0, 0.0)"
    sem = stats.sem(data)

    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean,
        scale=sem
    )

    return f"{mean:.2f} ({ci_low:.2f}, {ci_high:.2f})"

def main(args):
    data_path = f"{DATA_FOLDER}/{args.experiment}/phrases_no_exclude_{args.data}.jsonl"
    
    if args.method == "io":
        accuracy, failure_rate = run_io(args.model_name, 
               data_path=data_path, 
               thinking = args.thinking,
               n_return=args.n_path)
        
    if args.method == "tot":
        accuracy, failure_rate = run_tot(
            data_path=data_path,
            model_name=args.model_name, 
            thinking = args.thinking,
            n_return=args.n_path)
    
    return accuracy, failure_rate
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen3_8b")
    parser.add_argument("--n_path", type=int, default=1)
    parser.add_argument("--method", type=str, choices=["io", "tot"], default="io")
    parser.add_argument("--experiment", type=str, choices=["original", "na", "abstention"], default="original")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--data", type=str, choices=["dev", "testt"], default="dev")
    args = parser.parse_args()

    print(f"Starting {args.experiment} experiment with method: {args.method} using {args.model_name}")
    
    metrics = []
    failures = []
    for i in range(5):
        random.seed(2*(i^3) + 42)
        torch.manual_seed(2*(i^3) + 42)
        accuracy, failure_rate = main(args)
        metrics.append(accuracy)
        failures.append(failure_rate)
    
    acc_ci = mean_ci_95(metrics)
    fail_ci = mean_ci_95(failures)

    if args.thinking:
        name_tag = "thinking"
    else:
        name_tag = "no_thinking"

    with open(f"./data/{args.model_name}_{args.experiment}_{args.method}_{name_tag}_results.txt", "w") as file:
        file.write(f"Accuracy: {acc_ci}\n")
        file.write(f"Unparsed responses: {fail_ci}\n")
        file.write(f"The {args.experiment} experiment with method - {args.method} using {args.model_name} is finished\n")
        file.write("Results are shown above")
