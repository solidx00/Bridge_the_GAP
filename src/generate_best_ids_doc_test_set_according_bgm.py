import os
import re 
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Union, List

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
from trl import setup_chat_format

from utils import *
from bgm import BGM
from llm import LLM
from default_prompts import *
from prompt_dataset import PromptDataset
from datasets import Dataset

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
        'output_dir': r'C:\Users\franc\Documents\Bridge_the_GAP\data\gen_best_ids_doc_test_set_according_bgm',
        'bgm_id': 'meta-llama/Llama-3.2-1B',
        'llm_id': 'google/gemma-2-2b-it',
        'dataset': 'nq',
        'task_instruction' : "Output only the IDs of the documents relevant to the query. Use this format: [Id_1, Id_2, ...]. If there is no relevant document, output NO_DOCS.",
        'model_max_length': 4096,
        'quantization_bits': 4,
        'gold_position': None,
        'use_model_chat_template': True, 
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

        prompt_template = apply_chat_task_template(chat_task_template_str, task_instruction)
    else:
        task_template = task_templates[prompt_configuration]

        prompt_template = task_template.create_prompt_template()

    return prompt_template

def process_dataset(dataset, task_instruction):
    """
    Processes the dataset by applying the chat template transformation.

    Args:
        dataset (List[Dict]): The dataset to be processed.

    Returns:
        List[Dict]: The processed dataset with formatted text.
    """

    # Define the chat messages format
    messages = [
        {"role": "system", "content": task_instruction},
        {"role": "user", "content": dataset},
    ]

    return messages

def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    retriever_search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer,
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

    
def map_document_indices(prompt: str, document_indices: list) -> tuple:
    """
    Modifica il prompt mappando i document ID originali con ID sequenziali.
    
    Args:
        prompt (str): Il prompt originale contenente i document ID.
        document_indices (list): Lista degli indici dei documenti nel prompt.

    Returns:
        tuple: Il prompt modificato e la mappatura (original_id -> new_id).
    """
    id_mapping = {}
    
    # Mappa i document ID ai nuovi ID sequenziali (Id_1, Id_2, ...)
    for idx, doc_id in enumerate(document_indices, start=1):
        new_id = f"Id_{idx}"
        id_mapping[str(doc_id)] = new_id
    
    # Sostituisce i document ID nel prompt
    modified_prompt = prompt
    for original_id, new_id in id_mapping.items():
        modified_prompt = re.sub(rf'\[{original_id}\]', f'[{new_id}]', modified_prompt)
    
    return modified_prompt, id_mapping

def extract_and_convert_answer_indices(generated_output: str, id_mapping: dict) -> str:
    """
    Estrae e converte gli ID generati dal modello dopo '<|im_start|>assistant'.
    Converte gli ID nel formato originale. Se non trova un ID nella mappatura, restituisce 'Unknown(ID)'.

    Args:
        generated_output (str): Testo con la risposta generata dal modello.
        id_mapping (dict): Mappatura {original_id: Id_n}.

    Returns:
        str: Stringa con gli ID originali separati da virgola.
    """
    # Invertire la mappatura per ottenere {Id_n: original_id}
    inverse_mapping = {v: k for k, v in id_mapping.items()}

    # Estrae la risposta dopo '<|im_start|>assistant'
    match = re.search(r'<\|im_start\|>assistant\s*(.*)', generated_output, re.DOTALL)
    if not match:
        return "NO_DOCS"
    #if match:
        #print("Contenuto dopo assistant:", repr(match.group(1)))

    # Ottieni la stringa con gli ID dopo assistant
    answer_string = match.group(1).strip().split("<|im_end|>")[0].strip()

    # Dividi e converte gli ID
    #generated_ids = [id_.strip() for id_ in answer_string.split(",") if id_.strip()] #splitta solo con la ','
    generated_ids = [id_.strip() for id_ in re.split(r'[\s,]+', answer_string) if id_.strip()]  #splitta anche con gli spazi
    original_ids = [inverse_mapping.get(id_, f"Unknown({id_})") for id_ in generated_ids]

    # Restituisci gli ID originali come stringa separata da virgole
    return ",".join(original_ids)

def reconstruct_prompt_from_ids(
    original_ids: str,
    prompt: str,
    task_instruction: str = "You are given a question and you must respond based on the provided documents. You must always provide an answer."
) -> str:
    """
    Ricostruisce il prompt originale utilizzando solo i documenti selezionati dal modello,
    ignorando gli ID non presenti nella lista generale e rimuovendo le interruzioni tra documenti.

    Args:
        original_ids (str): Stringa contenente gli ID separati da virgola (es. "628506,3546609").
        prompt (str): Testo completo del prompt originale.

    Returns:
        str: Il prompt ricostruito contenente solo i documenti selezionati.
    """
    query_match = re.search(r"Question:\s*(.*?)(?=Document|$)", prompt, re.DOTALL)
    query = query_match.group(1).strip() if query_match else "Query non trovata"
    
    documents = re.findall(
        r"(Document \[\d+\]\(.*?\)\s.*?)(?=Document \[\d+\]|$)", 
        prompt, 
        re.DOTALL
    )
    
    documents_dict = {re.search(r"Document \[(\d+)\]", doc).group(1): doc.strip() for doc in documents}
    
    selected_ids = [id_.strip() for id_ in original_ids.split(",") if id_.strip()]
    valid_ids = [id_ for id_ in selected_ids if id_ in documents_dict]
    
    selected_documents = [
        re.sub(r"Answer:\s*\n?", "", re.sub(r"<end_of_turn>\s*<start_of_turn>model", "", documents_dict[id_]))
        for id_ in valid_ids
    ]
    
    if not selected_documents:
        final_prompt = (f"<start_of_turn>user\n{task_instruction}\nQuestion: {query}\n<end_of_turn>\n<start_of_turn>model")
        return final_prompt
    
    reconstructed_prompt = "\n".join(selected_documents)
    
    final_prompt = (f"<start_of_turn>user\n{task_instruction}\nQuestion: {query}\nDocuments:\n{reconstructed_prompt}\n<end_of_turn>\n<start_of_turn>model")
    
    final_prompt = re.sub(r"\n+", "\n", final_prompt).strip() + '\n'
    
    return final_prompt


def print_info(args: argparse.Namespace):
    print("INFO:")    
    print(f"DATA: {info[args.dataset]['test']['data_path']}")
    print(f"TASK INSTRUCTION: {args.task_instruction}")
    print(f"USE TEST: {args.use_test}")
    print(f"BGM MODEL: {args.bgm_id}")
    print(f"MODEL MAX LENGTH: {args.model_max_length}")
    print(f'MAX NEW TOKENS: {args.max_new_tokens}')
    print(f"USE MODEL CHAT TEMPLATE: {args.use_model_chat_template}")
    print(f"TASK WITH PROOF:", args.use_task_with_proof)
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_retrieved_documents}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def generate_and_save(
    args: argparse.Namespace, 
    model_weights_path, 
    tokenizer,
    dataset,
    steps_training: int,
    max_length=50
):
    
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

    json_filename = f"{saving_dir}/numdoc{num_doc}_retr{args.num_retrieved_documents}{padding_str}{chat_template_str}_bgm_training_steps_{steps_training}_info.json"

    # Load the trained model
    model = AutoModelForCausalLM.from_pretrained(model_weights_path)
    model.to(device)
    model.eval()

    all_info = [] 
    for idx, prompt_batch in enumerate(tqdm(dataset)):
            
        prompt = prompt_batch['prompt'].replace('You are given a question and you must respond based on the provided documents. You must always provide an answer.', "")
        document_indices= prompt_batch['document_indices']

        # Mappa i document ID nel prompt
        modified_prompt, id_mapping = map_document_indices(prompt, document_indices)

        prompt_formatted = process_dataset(modified_prompt, args.task_instruction)
        prompt_formatted = tokenizer.apply_chat_template(
            prompt_formatted, tokenize=False, add_generation_prompt=False, add_special_tokens=False
        )

        # Tokenize the input
        inputs = tokenizer(prompt_formatted, return_tensors="pt", truncation=False).to(device)

        # Generate the model's response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=5)

        # Decode the generated response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Convertire gli ID generati nei document ID originali
        original_ids = extract_and_convert_answer_indices(generated_text, id_mapping)

        filtered_prompt = reconstruct_prompt_from_ids(original_ids, prompt)

        prompt_batch['bgm_indices'] = original_ids
        
        prompt_batch['prompt'] = filtered_prompt
        
        all_info.append(prompt_batch)

    # Salvataggio finale in un unico file JSON
    print(f"Saving all data in a single JSON file: {json_filename}")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(all_info, f, ensure_ascii=False, indent=4)
        
        

def main():
    args = parse_arguments()

    args.split = "test" if args.use_test else "train"

    print("Loading BGM...")
    bgm_id = args.bgm_id
    
    bgm = BGM(
        bgm_id, device, 
        quantization_bits=args.quantization_bits, 
        model_max_length=args.model_max_length,
    )
    bgm_model= bgm.model

    tokenizer = bgm.tokenizer
    model, tokenizer = setup_chat_format(bgm_model, tokenizer)
    print("BGM loaded")

    task_instruction = args.task_instruction

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

    training_path=r'C:\Users\franc\Documents\Bridge_the_GAP\data\SFT_training_bgm\meta-llama-Llama-3.2-1B\checkpoint-1700'
    generate_and_save(args, training_path, tokenizer, prompt_dataloader, steps_training=1700, max_length=15)



if __name__ == "__main__":
    seed_everything(SEED)
    main()