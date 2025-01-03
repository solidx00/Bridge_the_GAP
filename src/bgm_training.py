import json
from utils import *
import warnings
from bgm_prompt_dataset import BGMPromptDataset
from bgm import BGM
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "nq": {
        "train": {
            "data_path": r"C:\Users\franc\Documents\Bridge_the_GAP\data\gen_ids_document_training_set_bgm\nq_training\gemma-2-2b-it\train\retrieved\contriever\5_doc\numdoc5_retr5_info_all_extended_training_set.json",
            "corpus_file": r"C:\Users\franc\Documents\Bridge_the_GAP\data\corpus_with_contriever_at150.json",
        },
    }
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
        'llm_id': 'google-t5/t5-base',
        'dataset': 'nq',
        'model_max_length': 4096,
        'quantization_bits': 4,
        'task_instruction' : "Output only the document IDs relevant to the query. Use this format: [ID1, ID2, ...].",
        'use_test': False,
        'padding_strategy': 'longest',
        'max_new_tokens': 15,
        'batch_size': None,
        'save_every': 250,
    }

    # If custom_args is provided, update defaults
    if custom_args:
        default_args.update(custom_args)

    return DotDict(**default_args)

def load_corpus(
        args: argparse.Namespace,
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    
    # Corpus with documents from Contriever
    corpus, full_to_subset_idx_map = read_corpus_with_contriever()

    return corpus, full_to_subset_idx_map

def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    tokenizer: PreTrainedTokenizer,
    task_instruction
) -> DataLoader:

    dataset = BGMPromptDataset(
        data_path=info[args.dataset][args.split]['data_path'],
        tokenizer=tokenizer,  
        max_tokenized_length=args.model_max_length - 2,
        task_instruction=task_instruction, 
        corpus=corpus, 
        full_to_subset_idx_map = full_to_subset_idx_map,)
        
    prompt_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return prompt_dataloader

def main():

    args = parse_arguments()

    args.split = "test" if args.use_test else "train"

    print("Loading LLM...")
    llm_id = args.llm_id

    bgm = BGM(
        llm_id, device,  
        model_max_length=args.model_max_length
    )

    tokenizer = bgm.tokenizer
    print("LLM loaded")

    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    print("Corpus and search results loaded")

    # Inizializza task instructions
    task_instruction = args.task_instruction

    tokenizer = bgm.tokenizer

    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, tokenizer, task_instruction)
    print("Prompt dataset loaded")
    
    #Debug Dataloader
    for i, entry in enumerate(prompt_dataloader.dataset[:5]):
        print(f"Esempio {i}: {entry}")


if __name__ == "__main__":
    seed_everything(SEED)
    main()