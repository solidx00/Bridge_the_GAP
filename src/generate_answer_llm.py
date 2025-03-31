import os
import re 
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from utils import *
from llm import LLM
from default_prompts import *
from prompt_dataset import PromptDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "nq": {
        "test": {
            "data_path": r'C:\Users\franc\Documents\Bridge_the_GAP\data\test_dataset.json',
            "contriever_search_results_path": r"C:\Users\franc\Documents\Bridge_the_GAP\data\processed\contriever_test_search_results_at150.pkl",
        }
    },
}

class DotDict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_arguments(custom_args=None):
    """
    Mimics argparse to parse arguments for LLM generation. Accepts custom arguments as a dictionary for notebooks.
    """
    # Define default values
    default_args = {
        'output_dir': r'C:\Users\franc\Documents\Bridge_the_GAP\data\gen_res_example_llm',
        'llm_id': 'google/gemma-2-2b-it',
        'dataset': 'nq',
        'model_max_length': 4096,
        'quantization_bits': 4,
        'use_model_chat_template': True, 
        'gold_position': None,
        'num_retrieved_documents': 5,
        'use_test': True,
        'padding_strategy': 'longest',
        'max_new_tokens': 50,
        'use_task_with_proof': False,
        'batch_size': None,
        'save_every': 250,
    }

    # If custom_args is provided, update defaults
    if custom_args:
        default_args.update(custom_args)

    # Perform validation
    if default_args['num_retrieved_documents'] is None:
        raise ValueError("'num_retrieved_documents' must be specified.")
    if default_args['num_retrieved_documents'] <= 0:
        raise ValueError("'num_retrieved_documents' must be a positive integer.")
    if default_args['gold_position'] is not None:
        if (default_args['gold_position'] < 0 or 
            default_args['gold_position'] >= default_args['num_retrieved_documents']):
            raise ValueError("'gold_position' must be within the range of 'num_retrieved_documents'.")

    return DotDict(**default_args)


def load_corpus(
    args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    
    # Corpus with documents from Contriever
    corpus, full_to_subset_idx_map = read_test_corpus_with_random_and_contriever()

    return corpus, full_to_subset_idx_map

def load_search_results(args: argparse.Namespace) -> List[Tuple[List[int], List[float]]]:

    search_results_path = info[args.dataset][args.split]['contriever_search_results_path']
    retriever_search_results = read_pickle(search_results_path)

    return retriever_search_results


def get_prompt_template(args: argparse.Namespace):
    prompt_configuration = args.dataset
    if args.use_model_chat_template:
        chat_task_template_str = chat_task_templates[args.llm_id]['template']
        
        task_instruction = task_instructions[prompt_configuration]
        if args.use_task_with_proof:
            task_instruction = task_instructions['qa_proof'][prompt_configuration]

        prompt_template = apply_chat_task_template(chat_task_template_str, task_instruction)
    else:
        task_template = task_templates[prompt_configuration]

        if args.use_task_with_proof:
            task_template = task_templates['qa_proof'][prompt_configuration]

        prompt_template = task_template.create_prompt_template()

    return prompt_template


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    retriever_search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer
) -> DataLoader:
    
    prompt_template = get_prompt_template(args)
    
    prompt_ds = PromptDataset(
        corpus=corpus, data_path=info[args.dataset][args.split]['data_path'], 
        tokenizer=tokenizer, 
        max_tokenized_length=args.model_max_length - 2, 
        search_results=retriever_search_results,
        prompt_template=prompt_template,
        full_to_subset_idx_map=full_to_subset_idx_map,
        do_normalize_query=True, 
        num_documents_in_context=args.num_retrieved_documents,
        gold_position=args.gold_position, # None in these experiments
    )
        
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return prompt_dataloader


def print_info(args: argparse.Namespace):
    print("INFO:")    
    print(f"DATA: {info[args.dataset]['test']['data_path']}")
    print(f"USE TEST: {args.use_test}")
    print(f"MODEL: {args.llm_id}")
    print(f"MODEL MAX LENGTH: {args.model_max_length}")
    print(f'MAX NEW TOKENS: {args.max_new_tokens}')
    print(f"USE MODEL CHAT TEMPLATE: {args.use_model_chat_template}")
    print(f"TASK WITH PROOF:", args.use_task_with_proof)
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_retrieved_documents}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def extract_generate_answers(
    args: argparse.Namespace, 
    generated_output: List[str]
) -> List[str]:
    answer_prefix = "Answer:"
    if args.use_model_chat_template:
        answer_prefix = re.escape(chat_task_templates[args.llm_id]['answer_prefix'])

    generated_answers = []
    for output in generated_output:
        matches = list(re.finditer(answer_prefix, output))
        match_idx = 0

        # When using the proof there is a one-shot example that already 
        # contains the string "Answer:". Thus, we should get the second (match_idx=1) match.
        if args.use_task_with_proof:
            match_idx = 1
            if args.use_model_chat_template and answer_prefix != "Answer:":
                match_idx = 0
 
        answer_end = matches[match_idx].end()
        response = output[answer_end:].strip()
        generated_answers.append(response)
    
    return generated_answers


def generate_and_save(
    args: argparse.Namespace, 
    llm: LLM, 
    prompt_dataloader: DataLoader
):
    # Info from arguments
    llm_id = args.llm_id
    num_doc = args.num_retrieved_documents
    save_every = args.save_every
    retriever_str = "contriever"
    padding_str = f"_{args.padding_strategy}{args.model_max_length}" if args.padding_strategy != "longest" else "" 
    chat_template_str = "_template" if args.use_model_chat_template else ""
    prompt_type = "retrieved_proof" if args.use_task_with_proof else "retrieved"

    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{args.dataset}/{llm_folder}/{args.split}/{prompt_type}/{retriever_str}/{num_doc}_doc"
    os.makedirs(saving_dir, exist_ok=True)

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        generated_output = llm.generate(
            prompts,
            padding_strategy=args.padding_strategy, 
            max_new_tokens=args.max_new_tokens
        )

        generated_answers = extract_generate_answers(args, generated_output)
        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)

        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}_retr{args.num_retrieved_documents}{padding_str}{chat_template_str}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            all_info = []

def main():
    args = parse_arguments()

    args.split = "test" if args.use_test else "train"

    print("Loading LLM...")
    llm_id = args.llm_id
    llm = LLM(
        llm_id, device, 
        quantization_bits=args.quantization_bits, 
        model_max_length=args.model_max_length
    )
    tokenizer = llm.tokenizer
    print("LLM loaded")


    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    retriever_search_results = load_search_results(args)
    print("Corpus and search results loaded")


    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, 
        retriever_search_results, tokenizer
    )
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader)



if __name__ == "__main__":
    seed_everything(SEED)
    main()