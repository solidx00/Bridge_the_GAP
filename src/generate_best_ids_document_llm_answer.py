import os
import argparse
import warnings
import re
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from utils import *
from llm import LLM
from default_prompts import *
from normalize_answers import *
from prompt_dataset import PromptDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {

    "nq": {
        "test": {
            "data_path": r'C:\Users\franc\Documents\Bridge_the_GAP\data\test_dataset.json',
            "contriever_search_results_path": r"C:\Users\franc\Documents\Bridge_the_GAP\data\processed\contriever_test_search_results_at150.pkl",
        },
    },
    "nq_training":{
        "train": {
            "data_path": r"C:\Users\franc\Documents\Bridge_the_GAP\data\10k_train_dataset.json",
            "contriever_search_results_path": r"C:\Users\franc\Documents\Bridge_the_GAP\data\processed\contriever_search_results_at150.pkl",
        }
    }
}



def save_dataloader_to_json(dataloader, output_file, num_examples=15):
    all_batches = []

    print("Saving DataLoader contents to JSON...")
    for idx, batch in enumerate(dataloader):
        if idx >= num_examples:  # Stop after saving the specified number of examples
            break

        batch_dict = {}
        for key, value in batch.items():
            # Convert tensors to lists for JSON serialization
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.tolist()
            else:
                batch_dict[key] = value
        all_batches.append(batch_dict)
    
    # Save the entire list of dictionaries to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_batches, f, ensure_ascii=False, indent=4)

    print(f"DataLoader contents saved to {output_file}")

class DotDict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_arguments(custom_args=None):
    """
    Mimics argparse to parse arguments for LLM generation. Accepts custom arguments as a dictionary for notebooks.
    """
    # Define default values
    default_args = {
        'output_dir': r'C:\Users\franc\Documents\Bridge_the_GAP\data\gen_ids_document_training_set_bgm',
        'llm_id': 'google/gemma-2-2b-it',
        'dataset': 'nq_training',
        'model_max_length': 4096,
        'quantization_bits': 4,
        'use_model_chat_template': False, 
        'gold_position': None,
        'num_retrieved_documents': 5,
        'use_test': False,
        'max_new_tokens': 50,
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
    corpus, full_to_subset_idx_map = read_corpus_with_contriever()

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


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    retriever_search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer
) -> Tuple[PromptDataset, DataLoader]:
    
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
    return prompt_ds, prompt_dataloader


def print_info(args: argparse.Namespace):
    print("INFO:")    
    print(f"DATA: {info[args.dataset][args.split]['data_path']}")
    print(f"USE TEST: {args.use_test}")
    print(f"MODEL: {args.llm_id}")
    print(f"MODEL MAX LENGTH: {args.model_max_length}")
    print(f'MAX NEW TOKENS: {args.max_new_tokens}')
    print(f"USE MODEL CHAT TEMPLATE: {args.use_model_chat_template}")
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
        if args.use_model_chat_template and answer_prefix != "Answer:":
            match_idx = 0
 
        answer_end = matches[match_idx].end()
        response = output[answer_end:].strip()
        generated_answers.append(response)
    
    return generated_answers


def are_answers_matching(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth in normalized_prediction:
            return True
    return False

def calculate_bleu(model_answer, correct_answers):
    """
    Calculate the maximum BLEU score between a model's answer and a list of correct answers.

    Parameters:
        correct_answers (list of str): A list of correct answers.
        model_answer (str): The model's generated response.

    Returns:
        float: The highest BLEU score among the comparisons.
    """
    # Tokenize the model's answer
    model_answer_tokens = model_answer.split()

    # Initialize smoothing function to avoid zero scores for short sequences
    smoothing_function = SmoothingFunction().method1

    # Calculate BLEU scores for each correct answer
    bleu_scores = [
        sentence_bleu([correct_answer.split()], model_answer_tokens, smoothing_function=smoothing_function)
        for correct_answer in correct_answers
    ]

    # Return the highest BLEU score
    return max(bleu_scores)

def evaluate_accuracy(args: argparse.Namespace, llm, candidate_prompt, answers: List[str], max_new_tokens=50):
    """
    Valuta la generazioner del modello.
    Per semplicità, controlla se la risposta generata dal modello matcha con quella dell'esempio.
    """
    generated_answer = llm.generate(candidate_prompt, max_new_tokens=max_new_tokens)
    response_text = extract_generate_answers(args, generated_answer)
    response_text = response_text[0].split('\n', 1)[0]
    accuracy_answer = are_answers_matching(response_text, answers) if answers else False

    return accuracy_answer, response_text

### DA QUI INIZIALIZZARE L'ALGORITMO DI GENERAZIONE ###
def generate_and_save(
    args: argparse.Namespace, 
    llm: LLM,
    prompt_ds: PromptDataset,
    df: pd.DataFrame,
    prompt_dataloader: DataLoader,
    start_example: int = 1500,
    max_examples: int = 3000  
):
    # Info from arguments
    max_new_tokens=args.max_new_tokens
    llm_id = args.llm_id
    num_doc = args.num_retrieved_documents
    save_every = args.save_every
    retriever_str = "contriever" 
    chat_template_str = "_template" if args.use_model_chat_template else ""
    prompt_type = "retrieved"

    

    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{args.dataset}/{llm_folder}/{args.split}/{prompt_type}/{retriever_str}/{num_doc}_doc"
    os.makedirs(saving_dir, exist_ok=True)

    ### ALGORITMO DI GENERAZIONE ### 
    all_info = [] 
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):

        if idx < start_example:
            continue

        if idx >= max_examples:
            break

        example_id = prompt_batch['example_id']
        prompts = prompt_batch['prompt']
        query = prompt_batch['query']
        document_indices=prompt_batch['document_indices']

        answers = df[df['example_id'].astype(str) == str(example_id)].answers.iloc[0]

        
        # Informazioni da salvare
        query_info = {
            "query": query,
            "document_indices": document_indices,
            "selected_documents": [],
            "generated_responses": []
        }
        
        # Inizializza la sequenza ottimizzata
        d_silv = []
        d_silv_are_answer = False  # Punteggio iniziale neutrale
        d_silv_score = 0
        
        best_candidate = None
        best_candidate_are_answer = False  # Mantieni il punteggio corrente come riferimento
        best_candidate_score = 0
        
        evaluated_no_document_case = False

        while True: 
            # Valuta il caso in cui non ci sono documenti nel contesto
            if not evaluated_no_document_case:
                evaluated_no_document_case = True
        
                # Crea il prompt senza documenti
                candidate_prompt = prompts
                if '\nAnswer:' not in candidate_prompt:
                    candidate_prompt += '\nAnswer:'
        
                # Verifica se la risposta matcha con quella corretta dell'esempio in questione
                cur_are_answer, generated_answer = evaluate_accuracy(args, llm, candidate_prompt, answers, max_new_tokens)

                cur_score = calculate_bleu(generated_answer, answers)

                # Salva i dettagli della generazione
                query_info["generated_responses"].append({
                    "candidate_documents": [],
                    "generated_answer": generated_answer,
                    "is_correct": cur_are_answer,
                    "bleu_score": cur_score
                })

                # Aggiorna il miglior candidato se il punteggio migliora
                if cur_are_answer == True: 
                    best_candidate = None  # Non c'è un documento specifico
                    best_candidate_are_answer = cur_are_answer
                    best_candidate_score = cur_score

                    # Aggiorna la sequenza di documenti selezionati
                    d_silv = []
                    d_silv_are_answer = cur_are_answer
                    d_silv_score = cur_score
                
                '''
                print(f'Documenti in d_silv: {d_silv}')
                print(f'Prompt corrente senza documenti: {candidate_prompt}')
                print(f'Le risposte corrette sono{answers}')
                print(f'Risposta generata dal modello senza documenti: {generated_answer}')
                print(f"La risposta generata senza documenti e' corretta?: {best_candidate_are_answer}")
                print(f"La risposta senza documenti ha questa accuracy: {best_candidate_score}")
                '''

            # Valuta i documenti non ancora selezionati
            for doc_idx in document_indices:

                if doc_idx in d_silv:
                    continue  # Salta i documenti già selezionati

                # Crea un candidato aggiungendo il documento alla sequenza corrente
                candidate_docs = d_silv + [doc_idx]

                formatted_docs, _ = prompt_ds._get_documents_from_indices(candidate_docs)

                candidate_prompt = prompts
                # Aggiungi i documenti solo se non sono vuoti
                if formatted_docs:
                    candidate_prompt += "\n".join(formatted_docs)  #concatena il prompt di ogni esempio con ogni documento in input in quel momento

                if '\nAnswer:' not in candidate_prompt:
                    candidate_prompt += '\nAnswer:'

                cur_are_answer, generated_answer = evaluate_accuracy(args, llm, candidate_prompt, answers, max_new_tokens)

                '''
                print(f'Documenti in d_silv: {d_silv}')
                print(f'Documenti candidati: {candidate_docs}')
                print(f'Prompt corrente: {candidate_prompt}')
                print(f'Le risposte corrette sono{answers}')
                print(f'Risposta generata dal modello: {generated_answer}')
                print(f"Ce' la risposta corretta in quella generata dal modello? : {cur_are_answer}")
                '''

                cur_score = calculate_bleu(generated_answer, answers)

                query_info["generated_responses"].append({
                    "candidate_documents": candidate_docs,
                    "generated_answer": generated_answer,
                    "is_correct": cur_are_answer,
                    "bleu_score": cur_score
                })

                # Aggiorna il miglior candidato se il punteggio migliora
                if cur_are_answer == True: 
                    if best_candidate_are_answer == True:
                    
                        if cur_score > best_candidate_score:
                            best_candidate = doc_idx
                            best_candidate_are_answer = cur_are_answer
                            best_candidate_score = cur_score

                            #print(f"La risposta con questo documento {best_candidate} c'e'? {best_candidate_are_answer} e ha questa accuracy {best_candidate_score}")
                
                    else: 
                        best_candidate = doc_idx
                        best_candidate_are_answer = cur_are_answer
                        best_candidate_score = cur_score
                        #print(f"La risposta con questo documento {best_candidate} c'e'? {best_candidate_are_answer} e ha questa accuracy {best_candidate_score}\n")
                
                #else:
                    #print(f"La risposta con il documento {candidate_docs} non e'stata trovata\n")

            
            # Aggiungi il miglior candidato alla sequenza se migliora il punteggio
            if best_candidate is not None and best_candidate not in d_silv and best_candidate_are_answer == True:

                if d_silv_are_answer == False:
                    d_silv.append(best_candidate)
                    d_silv_are_answer = best_candidate_are_answer
                    d_silv_score = best_candidate_score

                    #print(f"\nSono nel secondo If e la risposta con questi documenti {d_silv} c'e'? {d_silv_are_answer} e ha questa accuracy {d_silv_score}\n")

                else:
                    if best_candidate_score > d_silv_score:
                        d_silv.append(best_candidate)
                        d_silv_score = best_candidate_score
                        d_silv_are_answer = best_candidate_are_answer
                    
                    #print(f"\nSono nel secondo If e la risposta con questi documenti {d_silv} c'e'? {d_silv_are_answer} e ha questa accuracy {d_silv_score}\n")
                
            else:
                #print(f"\nNon c'e' piu' nessun documento che migliora la risposta, la risposta migliore e' con questi documenti {d_silv} e ha questa accuracy {d_silv_score}\n")
                break  # Interrompi se nessun documento migliora il punteggio
        
        # Aggiorna i documenti selezionati finali
        query_info["selected_documents"] = d_silv

        all_info.append(query_info)
            
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}_retr{args.num_retrieved_documents}{chat_template_str}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            all_info = []
            


def main():
    args = parse_arguments()

    args.split = "test" if args.use_test else "train"

    print("Loading LLM...")
    llm_id = args.llm_id
    llm = LLM(
        llm_id, device,  
        model_max_length=args.model_max_length
    )
    tokenizer = llm.tokenizer
    print("LLM loaded")


    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    retriever_search_results = load_search_results(args)
    print("Corpus and search results loaded")


    print("Loading prompt dataset...")
    prompt_ds, prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, 
        retriever_search_results, tokenizer
    )
    print("Prompt dataset loaded")

    print_info(args)

    df = pd.read_json(info[args.dataset][args.split]['data_path'], dtype={'example_id': str})
    
    #output_json_path = r'C:\Users\franc\Documents\Bridge_the_GAP\data\dataloader_contents.json'
    #save_dataloader_to_json(prompt_dataloader, output_json_path, num_examples=15)
        
    generate_and_save(args, llm, prompt_ds, df, prompt_dataloader, 1500, 3000)



if __name__ == "__main__":
    seed_everything(SEED)
    main()