import os
import re
import json
import pickle
import argparse

import torch
import pandas as pd
from typing import List, Dict

from utils import str2bool
from normalize_answers import *
from read_negative_rejection import *


def are_answers_matching(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth in normalized_prediction:
            return True
    return False


def extract_proof_from_text(text: str) -> str:
    matches = list(re.finditer("Proof:", text))
    
    if matches:
        proof_end = matches[0].end()
        proof = text[proof_end:].strip()
        # Get the text until the first new line
        proof = proof.split('\n', 1)[0] 
    else:
        proof = "NO-PROOF"

    return proof


def compute_df_accuracy(df: pd.DataFrame, attribute: str) -> float:
    return round(df[attribute].sum() / len(df), 4) * 100

def contains_no_res(text: str) -> bool:
    NO_RES_PATTERNS = [ 
        "NO-RES",
        "any of the provided document",
        "any of the document",
        "None of the provided document",
        "None of the document",
        "No res",
        "NO RESULT FOUND",
        "NO RESPONSE",
        "NO-RESPONSE",
        "No answer found",
        "do not know the answer",
        "don't know the answer",
        "do not provide information",
        "does not provide any information",
        "no direct answer",
        "not explicitly stated in the given document",
    ]
    return any(pattern.lower() in text.lower() for pattern in NO_RES_PATTERNS)

def compute_custom_accuracy(df: pd.DataFrame) -> float:
    correct_count = 0

    for _, row in df.iterrows():
        ans_in_documents = row['ans_in_documents']
        generated_answer = row['generated_answer']
        ans_match_after_norm = row['ans_match_after_norm']

        if (ans_in_documents and ans_match_after_norm):
            correct_count += 1
        elif (not ans_in_documents and contains_no_res(generated_answer)):
            correct_count += 1
        # all other cases are incorrect, do nothing

    return round(correct_count / len(df), 4) * 100


def read_generation_results(file_path: str, df: pd.DataFrame) -> List[Dict]:
    data = []
    with open(file_path, "r") as fin:
        file_data = json.load(fin)

        # Handle list-based or dict-based structures
        if isinstance(file_data, list):
            examples = file_data
        else:
            examples = [file_data]  # Wrap single dictionary in a list for consistency

        for example in examples:
            example_id = example.get('example_id', [])
            query = example.get('query', [])
            prompt = example.get('prompt', [])
            document_indices = example.get('document_indices', [])
            gold_document_idx = example.get('gold_document_idx', [])
            generated_answer = example.get('generated_answer', [])
            prompt_tokens_len = example.get('prompt_tokens_len', [])

            documents_idx = list(document_indices) if isinstance(document_indices, list) else [document_indices]
            generated_answer = generated_answer[0].split('\n', 1)[0]
            filtered_df = df[df['example_id'].astype(str) == str(example_id)]
            if filtered_df.empty:
                print(f"Warning: No matching entry found for example_id {example_id}")
                answers = None
            else:
                answers = filtered_df.answers.iloc[0]

            gold_in_retrieved = int(gold_document_idx) in map(int, documents_idx)
            ans_match_after_norm = are_answers_matching(generated_answer, answers) if answers else False
            ans_in_documents = is_answer_in_text(prompt, answers) if answers else False

            # Modifica per gestire i casi NO-RESPONSE
            is_no_res = contains_no_res(generated_answer)
            is_correct = (ans_in_documents and ans_match_after_norm) or (not ans_in_documents and is_no_res)

            data.append({
                'example_id': str(example_id),
                'query': query,
                'prompt': prompt,
                'document_indices': documents_idx,
                'gold_document_idx': gold_document_idx,
                'generated_answer': generated_answer,
                'answers': answers,
                'ans_match_after_norm': ans_match_after_norm,
                'gold_in_retrieved': gold_in_retrieved,
                'ans_in_documents': ans_in_documents,
                'is_no_res': is_no_res,
                'is_correct': is_correct,
                "prompt_tokens_len": prompt_tokens_len,
            })

            if 'proof' in file_path:
                proof = extract_proof_from_text(generated_answer)
                data[-1]['proof'] = proof
                data[-1]['ans_in_proof'] = is_answer_in_text(proof, [generated_answer])

    return data

def read_generation_results_only_query(file_path: str, df: pd.DataFrame) -> List[Dict]:
    data = []
    with open(file_path, "r") as fin:
        file_data = json.load(fin)

        # Handle list-based or dict-based structures
        if isinstance(file_data, list):
            examples = file_data
        else:
            examples = [file_data]  # Wrap single dictionary in a list for consistency
        
        for example in examples:
            example_id = example.get('example_id', [])
            query = example.get('query', [])
            prompt = example.get('prompt', [])
            generated_answer = example.get('generated_answer', [])

            generated_answer = generated_answer[0].split('\n', 1)[0]
            
            answers = df[df['example_id'].astype(str) == str(example_id)].answers.iloc[0]

            ans_match_after_norm: bool = are_answers_matching(generated_answer, answers)
            ans_in_documents: bool = is_answer_in_text(prompt, answers)

            data.append({
                    'example_id': str(example_id),
                    'query': query,
                    'prompt': prompt,
                    'generated_answer': generated_answer,
                    'answers': answers,
                    'ans_match_after_norm': ans_match_after_norm,
                    'ans_in_documents': ans_in_documents,
                })        
    return data


def convert_tensors(cell):
    """ Converts tensors in the given cell to lists, if they are tensors. """
    if isinstance(cell, list):
        return [[t.tolist() if torch.is_tensor(t) else t for t in inner_list] for inner_list in cell]
    return cell


def extract_number_from_filename(filename: str, pattern: re.Pattern) -> int:
    """ Extracts the number from the filename based on the provided pattern. """
    match = pattern.search(filename)
    return int(match.group(1)) if match else 0


def load_pickle_files(directory: str, filename_prefix: str) -> pd.DataFrame:
    """ Loads and concatenates data from all pickle files in the directory with the given prefix. """
    pattern = re.compile(r'(\d+).pkl')
    files = [f for f in os.listdir(directory) if f.endswith('.pkl') and filename_prefix in f]
    files.sort(key=lambda f: extract_number_from_filename(f, pattern))
    print("I'm using the following files: ", files)

    data_list = []
    for file in files:
        with open(os.path.join(directory, file), 'rb') as f:
            data = pickle.load(f)
            data_list.extend(data)
    
    data_df = pd.DataFrame(data_list)
    if 'only_query' in directory:
        if data_df['example_id'].dtype != "O":
            data_df['example_id'] = data_df['example_id'].apply(lambda x: x.tolist())
    
    '''
    else:
        print(type(data_df['document_indices'].values))
        if not isinstance(data_df['document_indices'], list):
            data_df['document_indices'] = data_df['document_indices'].apply(convert_tensors)
    

    if 'prompt_tokens_len' in data_df.columns:
        data_df['prompt_tokens_len'] = data_df['prompt_tokens_len'].apply(lambda x: x.tolist())

    '''
        
    return data_df


def save_data_to_json(data_df: pd.DataFrame, directory: str, filename_prefix: str):
    """ Saves the given DataFrame to a JSON file. """
    data_path = os.path.join(directory, f'{filename_prefix}all.json')
    # Check if the file already exists
    if os.path.exists(data_path):
        overwrite = input(f"File {data_path} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("No overwrite.")

            results_df = pd.read_json(f'{directory}/{filename_prefix}all_extended.json')
            accuracy = compute_df_accuracy(results_df, 'ans_match_after_norm')
            print("ACCURACY: ", accuracy)

            if 'proof' in directory:
                accuracy_ans_in_proof = compute_df_accuracy(results_df, 'ans_in_proof')
                print("ACCURACY ANS IN PROOF", accuracy_ans_in_proof)

            correct_ans_not_in_context_accuracy = compute_accuracy_correct_answer_not_in_context(results_df)
            print(f"Correct Answer Not in Context Accuracy: {correct_ans_not_in_context_accuracy}")

            return None
        
    data_df.to_json(data_path, orient='records')
    return data_path


def get_retrieved_path(args):
    padding_str = f"_{args.padding_strategy}{args.model_max_length}" if args.padding_strategy != "longest" else "" 
    chat_template_str = "_template" if args.use_model_chat_template else ""

    filename_prefix = f"numdoc{args.num_doc}_retr{args.num_retrieved_documents}{padding_str}{chat_template_str}_info_"
    return filename_prefix


def get_only_query_path(args):
    chat_template_str = "_template" if args.use_model_chat_template else ""

    filename_prefix = f"only_query{chat_template_str}_info_"
    return filename_prefix

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
        'use_model_chat_template': True, 
        'gold_position': None,
        'num_retrieved_documents': 5,
        'use_test': True,
        'padding_strategy': 'longest',
        'max_new_tokens': 50,
        'prompt_type': 'retrieved'
    }

    # If custom_args is provided, update defaults
    if custom_args:
        default_args.update(custom_args)

    # Perform validation
    if not default_args['prompt_type'] in ['retrieved', 'retrieved_proof', 'only_query']:
        raise ValueError("Invalid prompt type. Must be one of ['retrieved', 'retrieved_proof', 'only_query']")
    
    return DotDict(**default_args)


info = {
    "nq": {
        "test": r'C:\Users\franc\Documents\Bridge_the_GAP\data\test_dataset.json',
    },
}

def main():
    args = parse_arguments()

    prompt_type = args.prompt_type
    
    if prompt_type == 'only_query':
        retriever_str=""
    else: 
        retriever_str = "contriever/"

    if 'retrieved' in prompt_type:    
        args.num_doc = args.num_retrieved_documents
        filename_prefix = get_retrieved_path(args)
    elif prompt_type == 'only_query':
        filename_prefix = get_only_query_path(args)
    else:
        raise ValueError("Invalid prompt type")


    llm_id = args.llm_id
    split = "test" if args.use_test else "train"
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    doc_str = f"{args.num_doc}_doc" if 'only_query' not in prompt_type else ""
    directory = f'{args.output_dir}/{args.dataset}/{llm_folder}/{split}/{prompt_type}/{retriever_str}{doc_str}'
    print("Directory: ", directory)

    df = pd.read_json(info[args.dataset][split], dtype={'example_id': str})

    data_df = load_pickle_files(directory, filename_prefix)
    data_path = save_data_to_json(data_df, directory, filename_prefix)
    if data_path is None:
        return
    
    if 'only_query' in directory:
        results = read_generation_results_only_query(data_path, df)
    else:
        results = read_generation_results(data_path, df)

    results_df = pd.DataFrame(results)

    #accuracy = compute_df_accuracy(results_df, 'ans_match_after_norm')
    #print("ACCURACY: ", accuracy)

    # New accuracy computing
    accuracy = compute_custom_accuracy(results_df)
    print("CUSTOM ACCURACY: ", accuracy)
    
    if 'proof' in directory:
        accuracy_ans_in_proof = compute_df_accuracy(results_df, 'ans_in_proof')
        print("ACCURACY ANS IN PROOF", accuracy_ans_in_proof)
        
    results_df.to_json(os.path.join(directory, f'{filename_prefix}all_extended.json'), orient='records')

    correct_ans_not_in_context_accuracy = compute_accuracy_correct_answer_not_in_context(results_df)
    print(f"Correct Answer Not in Context Accuracy: {correct_ans_not_in_context_accuracy}")

if __name__ == "__main__":
    main()