import json
from utils import *
import warnings
from tqdm import tqdm
from bgm_prompt_dataset import BGMPromptDataset
from bgm import BGM
from peft import LoftQConfig,LoraConfig, get_peft_model
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, LlamaConfig, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "nq": {
        "train": {
            "data_path": r"C:\Users\franc\Documents\Bridge_the_GAP\data\gen_best_ids_document_training_set_bgm\nq_training\gemma-2-2b-it\train\retrieved\contriever\5_doc\numdoc5_retr5_info_all_extended_training_set.json",
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

    default_args = {
        'output_dir': r'C:\Users\franc\Documents\Bridge_the_GAP\data\lora_training_bgm\lora-checkpoint',
        'llm_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'dataset': 'nq',
        'model_max_length': 4096,
        'quantization_bits': 4,
        'task_instruction' : "Output only the document IDs relevant to the query. Use this format: [Id_1, Id_2, ...].",
        'use_test': False,
        'padding_strategy': 'max_length',
        'max_new_tokens': 15,
        'batch_size': None,
        'save_every': 250,
    }

    if custom_args:
        default_args.update(custom_args)

    return DotDict(**default_args)

def print_info(args: argparse.Namespace):
    print("INFO:")    
    print(f"DATA: {info[args.dataset][args.split]['data_path']}")
    print(f"TASK INSTRUCTION: {args.task_instruction}")
    print(f"USE TEST: {args.use_test}")
    print(f"MODEL: {args.llm_id}")
    print(f"MODEL MAX LENGTH: {args.model_max_length}")
    print(f'MAX NEW TOKENS: {args.max_new_tokens}')
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")

def load_corpus(
        args: argparse.Namespace,
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    
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

def pad_to_nearest_multiple(inputs, n_heads, tokenizer):
    """
    Pads the sequence length to the nearest multiple of n_heads.
    
    Args:
        inputs (dict): The tokenized inputs with keys "input_ids" and "attention_mask".
        n_heads (int): The number of attention heads in the model.
        tokenizer: The tokenizer used for padding.

    Returns:
        dict: The updated inputs with padded sequences.
    """
    seq_len = inputs["input_ids"].size(1)
    padding_needed = n_heads - (seq_len % n_heads) if seq_len % n_heads != 0 else 0

    if padding_needed > 0:
        # Pad input_ids with pad_token_id
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"],
            (0, padding_needed),
            value=tokenizer.pad_token_id
        )
        # Pad attention_mask with 0 (no attention on padding tokens)
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"],
            (0, padding_needed),
            value=0
        )

    return inputs


def bgm_training(args, bgm, prompt_dataloader, num_epochs=3, learning_rate=1e-4, patience=3, min_delta=1e-4):
    """
    Function to train BGM model utilizing LoRA.
    
    Args:
        args: Configuration parameters.
        bgm: BGM object, which includes the model and the tokenizer.
        prompt_dataloader: DataLoader for the prompt dataset.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs without improvement before stopping the training.
        min_delta (float): Minimum improvement in loss required to be considered significant.
    """

    print("LoRA configuration...")
    
    # Controllo del modello basato sul nome o tipo
    if "gemma" in args.llm_id.lower():
        print("Configuring LoRA for Gemma model...")
        lora_config = LoraConfig(
            r=8,  # Grado della decomposizione low-rank
            lora_alpha=32,
            target_modules=["q", "v"],  
            lora_dropout=0.1,
            bias="none"  
    )
    else:
        print(f"Configuring LoRA for generic model: {args.llm_id}")
        lora_config = LoraConfig(
            r=8,  # Different rank for generic models
            lora_alpha=32,
            target_modules=["self_attn.q_proj", "self_attn.v_proj"],
            lora_dropout=0.1,
            bias="none"
        )

    # LoRA
    peft_model = get_peft_model(bgm.model, lora_config).to(device)

    print("LoRA model configured.")

    optimizer = AdamW(peft_model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_loss = float("inf")
    patience_counter = 0
    
    peft_model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(prompt_dataloader, desc="Batch Progress")):

            optimizer.zero_grad()
            
            inputs = batch["input"]["prompt"]
            labels = batch["output"]

            # Handle empty labels
            labels = batch["output"]
            if not labels:
                labels = [""]
            
            inputs = bgm.tokenizer(inputs, padding=False, return_tensors="pt", max_length=args.model_max_length, truncation=True).to(device)
            input_lenght = inputs["input_ids"].shape[1]
            labels = bgm.tokenizer(labels, padding=args.padding_strategy, return_tensors="pt", max_length=input_lenght, truncation=True).to(device)
            labels["input_ids"][labels["input_ids"] == bgm.tokenizer.pad_token_id] = -100

            # Ensure sequence length is compatible with attention heads
            if "gemma" in args.llm_id.lower():
                n_heads = peft_model.config.num_heads
                inputs = pad_to_nearest_multiple(inputs, n_heads, bgm.tokenizer)

            outputs = peft_model(
                **inputs, 
                labels=labels["input_ids"]
                )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        
        # Calcola la perdita media per epoca
        mean_loss = total_loss / len(prompt_dataloader)
        print(f"Epoch {epoch + 1} complete. Mean Loss: {mean_loss:.4f}")

        # Early stopping logic
        if mean_loss < best_loss - min_delta:
            best_loss = mean_loss
            patience_counter = 0
            # Salvataggio checkpoint del modello migliore
            output_dir = f"{args.output_dir}/{args.llm_id}/best_checkpoint"
            peft_model.save_pretrained(output_dir)
            print(f"Checkpoint saved in {output_dir} (Best Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement in loss. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break
        
        output_dir = f"{args.output_dir}/{args.llm_id}/epochs/epoch_{epoch + 1}"
        peft_model.save_pretrained(output_dir)
        print(f"Checkpoint saved in {output_dir}")
    

    print("Training complete.")
    return peft_model

    

def main():

    args = parse_arguments()

    args.split = "test" if args.use_test else "train"

    print("Loading LLM...")
    llm_id = args.llm_id

    bgm = BGM(
        llm_id, device, quantization_bits=args.quantization_bits,  
        model_max_length=args.model_max_length
    )

    tokenizer = bgm.tokenizer
    print("LLM loaded")

    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    print("Corpus and search results loaded")

    task_instruction = args.task_instruction

    tokenizer = bgm.tokenizer

    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, tokenizer, task_instruction)
    print("Prompt dataset loaded")

    #print("Printing 5 examples from the prompt dataloader...")
    #for i, example in enumerate(prompt_dataloader):
        #print(f"Example {i + 1}: {example}")
        #if i == 10:
            #break

    bgm_training(args, bgm, prompt_dataloader, num_epochs=50, patience=3, min_delta=1e-4)


if __name__ == "__main__":
    seed_everything(SEED)
    main()