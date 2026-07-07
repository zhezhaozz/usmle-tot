import argparse
import os
os.environ['HF_HOME'] = '/nfs/turbo/umms-vgvinodv/users/zzhaozhe/cache'

from src.io import run_io
from src.input_utils import DATA_FOLDER

def main(args):
    data_path = f"{DATA_FOLDER}/{args.experiment}/phrases_no_exclude_{args.data}.jsonl"
    
    print(f"Starting {args.experiment} experiment with method: {args.method} using {args.model_name}")

    if args.method == "io":
        run_io(args.model_name, 
               data_path=data_path, 
               thinking = args.thinking,
               n_return=args.n_path)
    
    print(f"The {args.experiment} experiment with method - {args.method} using {args.model_name} is finished")
    print("Results are shown above")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen3_8b")
    parser.add_argument("--n_path", type=int, default=1)
    parser.add_argument("--method", type=str, choices=["io", "tot"], default="io")
    parser.add_argument("--experiment", type=str, choices=["original", "na", "abstention"], default="original")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--data", type=str, choices=["dev", "testt"], default="dev")
    args = parser.parse_args()
    main(args)
