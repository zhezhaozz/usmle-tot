import os
os.environ['HF_HOME'] = '/nfs/turbo/umms-vgvinodv/users/zzhaozhe/cache'

from tqdm import tqdm
from collections import Counter
from itertools import chain

from src.configure_llm import *
from src.input_utils import load_dataset, construct_prompt
from src.prompts import tot_evaluation_prompt,tot_generation_prompt,tot_reasoning_generation_prompt,tot_termination_prompt
from src.output_utils import parse_json_output
from src.evaluate import evaluate


def get_votes(all_prediction, votes):
    """
    Get vote counts for EACH note 
    """
    #if "invalid" in all_prediction:
    #    if len(all_prediction) != len(votes.split(",")):
    #        all_prediction = all_prediction*len(votes.split(","))

    vote_num = []
    for v in votes.split(","):
        vote_num.append(int(v.split(":")[-1].replace("}", "").strip()))

    c = Counter()
    for lb, n in zip(all_prediction,vote_num):
        if lb == "invalid":
            break
        c[lb] += n
    
    return c

def most_frequent_item(lst):
    # Count the frequency of each element
    counts = Counter(lst)
    
    # Handle the case of an empty list
    if not counts:
        return None

    # Get the items with the highest frequency
    max_freq = max(counts.values())
    items_with_max_freq = [item for item, freq in counts.items() if freq == max_freq]
    
    # If there's only one such item, return it
    return items_with_max_freq[0]

class tot_classifier:
    def __init__(self,
                 note,
                 model_name,
                 thinking=False, 
                 do_sample = True, 
                 n_gen_path = 5,
                 n_vote = 5,
                 n_choose = 1):
        self.note = note
        self.model_name = model_name
        self.thinking = thinking
        self.do_sample = do_sample
        self.n_gen_path = n_gen_path
        self.n_vote = n_vote
        self.n_choose = n_choose

    def plan_generator(self, text): 
        chat = construct_prompt(tot_generation_prompt, text, mode="tot_generation")
        output = prompt_model(self.model_name, chat, self.n_gen_path, thinking=self.thinking)
        output = [ot["content"] for ot in output]
        return output

    def reasoning_generator(self, text, reasoning_steps = None):
        chat = construct_prompt(tot_reasoning_generation_prompt, text, mode="tot_reasoning_generation", tot_add=reasoning_steps)
        output = prompt_model(self.model_name, chat, self.n_gen_path, thinking=self.thinking)
        output = [ot["content"] for ot in output]
        return output
    
    def evaluator(self, text, mode, thoughts=None):
        chat = construct_prompt(tot_evaluation_prompt, text, mode, tot_add=thoughts)
        output = prompt_model(self.model_name, chat, self.n_vote, thinking=self.thinking)
        output = [ot["content"] for ot in output]
        return output
        

    def terminator(self, text, thoughts=None):
        chat = construct_prompt(tot_termination_prompt, text, mode="tot_termination", tot_add=thoughts)
        output = prompt_model(self.model_name, chat, 1, thinking=self.thinking)
        return output["content"]
    
    def propose_reasoning(self, proposed_step):
        reasoning_output = self.reasoning_generator(self.note, proposed_step)
        proposed_reasonings = [parse_json_output(ot, mode="tot_reasoning_generation") for ot in reasoning_output]

        return proposed_reasonings

    def propose_step(self):
        output = self.plan_generator(self.note)
        proposed_steps = [parse_json_output(ot, mode="tot_plan_generation") for ot in output]
        proposed_steps = [pt for pt in proposed_steps if type(pt) != int]

        return proposed_steps
    
    def create_candidates_set(self, proposals):
        thoughts = ''
        for i, p in enumerate(proposals, 1):
            thoughts += f'{i}: {p}\n'

        return thoughts

    def run(self):
        # sampling thoughts
        proposals = self.propose_step() 

        #remove duplicated options
        proposals = list(dict.fromkeys(proposals))

        candidate_thoughts = self.create_candidates_set(proposals)

        # evaluate the steps
        eval_output = self.evaluator(text=self.note, mode="tot_evaluation1", thoughts=candidate_thoughts)
        step_votes = [parse_json_output(eo, mode='tot_evaluation') for eo in eval_output]
        step_votes = Counter(step_votes)
        try:
            chosen_thoughts = [proposals[v-1] for v, count in step_votes.most_common(self.n_choose)]
            chosen_votes = Counter()
            for v, count in step_votes.most_common(self.n_choose):
                chosen_votes[v] = count
        except:
            return None

        # generate reasoning
        reasonings = list(chain(*[self.propose_reasoning(ps) for ps in chosen_thoughts]))

        #remove duplicated options
        reasonings = list(dict.fromkeys(reasonings))

        candidate_reasonings = self.create_candidates_set(reasonings)

        # evaluate and choose best reasonings
        eval_output = self.evaluator(text=self.note, mode="tot_evaluation2", thoughts=candidate_reasonings)
        votes = [parse_json_output(eo, mode='tot_evaluation') for eo in eval_output]
        votes = Counter(votes)
        try:
            chosen_reasonings = [reasonings[v-1] for v, count in votes.most_common(self.n_choose)]
            chosen_votes = Counter()
            for v, count in votes.most_common(self.n_choose):
                chosen_votes[v] = count
        except:
            return None

        # check termination
        answer = parse_json_output(self.terminator(self.note, thoughts=chosen_reasonings[-1]), mode='tot_termination')
        
        if answer:
            return answer
        else:
            return None


def run_tot(model_name, 
            data_path,
            thinking, 
            do_sample=True,
            n_return=5,
            n_eval=5):
    # load dataset
    dataset = load_dataset(data_path)

    # construct model inputs
    results = []
    for dt in tqdm(dataset):
        classifier = tot_classifier(
            dt,
            model_name,
            thinking=thinking,
            do_sample=do_sample,
            n_gen_path=n_return,
            n_vote=n_eval,
            n_choose=1
        )
        pred = classifier.run()
        results.append(pred)
    
    accuracy, failure_rate = evaluate(dataset, results)
    return accuracy, failure_rate


    
