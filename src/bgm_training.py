import json
from utils import *
import warnings
from tqdm import tqdm
from bgm_prompt_dataset import BGMPromptDataset
from bgm import BGM
from peft import LoraConfig
from transformers import TrainingArguments, pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from trl import setup_chat_format, SFTTrainer, SFTConfig
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
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
        'output_dir': r'C:\Users\franc\Documents\Bridge_the_GAP\data\SFT_training_bgm',
        'llm_id': 'meta-llama/Llama-3.2-1B',
        'dataset': 'nq',
        'model_max_length': 4096,
        'quantization_bits': 4,
        'task_instruction' : "Output only the document IDs relevant to the query. Use this format: [Id_1, Id_2, ...].",
        'use_test': False,
        'padding_strategy': 'longest',
        'max_new_tokens': 15,
        'batch_size': None,
        'save_every': 250,
    }

    # Generate the output directory dynamically using llm_id
    llm_id_cleaned = default_args['llm_id'].replace("/", "-")
    default_args['output_dir'] = rf"C:\Users\franc\Documents\Bridge_the_GAP\data\SFT_training_bgm\{llm_id_cleaned}"

    if custom_args:
        default_args.update(custom_args)

        # Ensure output_dir dynamically updates if llm_id is changed in custom_args
        if 'llm_id' in custom_args:
            llm_id_cleaned = custom_args['llm_id'].replace("/", "-")
            default_args['output_dir'] = rf"C:\Users\franc\Documents\Bridge_the_GAP\data\SFT_training_bgm\{llm_id_cleaned}"

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

def process_dataset_training(dataset, task_instruction):
    """
    Processes the dataset by applying the chat template transformation.

    Args:
        dataset (List[Dict]): The dataset to be processed.

    Returns:
        List[Dict]: The processed dataset with formatted text.
    """
    processed_dataset = []

    for sample in dataset:
        prompt = sample['input']['prompt']
        output = sample['output']
        
        # Define the chat messages format
        messages = [
            {"role": "system", "content": task_instruction},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]
        
        processed_dataset.append(messages)

    return processed_dataset

def process_dataset_testing(dataset, task_instruction):
    """
    Processes the dataset by applying the chat template transformation.

    Args:
        dataset (List[Dict]): The dataset to be processed.

    Returns:
        List[Dict]: The processed dataset with formatted text.
    """
    processed_dataset = []

    for sample in dataset:
        prompt = sample['input']['prompt']
        output = sample['output']
        
        # Define the chat messages format
        messages = [
            {"role": "system", "content": task_instruction},
            {"role": "user", "content": prompt},
        ]
        
        processed_dataset.append(messages)

    return processed_dataset

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
        #task_instruction=task_instruction, 
        corpus=corpus, 
        full_to_subset_idx_map = full_to_subset_idx_map,)
    
    dataset_training = process_dataset_training(dataset, task_instruction)
    dataset_testing = process_dataset_testing(dataset, task_instruction)

    dataset_training = Dataset.from_dict({"chat": dataset_training})
    dataset_testing = Dataset.from_dict({"chat": dataset_testing})

    dataset_training = dataset_training.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False, add_special_tokens=False)})

    dataset_testing = dataset_testing.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False, add_special_tokens=False)})

    return dataset_training, dataset_testing



def bgm_training(args, model, tokenizer, dataset):
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

    print("SFTT configuration...")
    
    
    sft_config = SFTConfig(
        dataset_text_field="formatted_chat",  # Specify the field containing the conversation
        output_dir=args.output_dir,
        max_steps=1000,  # Adjust based on dataset size and desired training duration
        per_device_train_batch_size=4,  # Set according to your GPU memory capacity
        learning_rate=5e-5,  # Common starting point for fine-tuning
        logging_steps=10,  # Frequency of logging training metrics
        save_steps=100,  # Frequency of saving model checkpoints
    )

    print("LoRA model configured.")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("Training complete.")

    # save model
    trainer.save_model()
    print("Training weights saved.")

def evaluate_model(model_weights_path, tokenizer, dataset, num_examples=10, max_length=50):
    """
    Generate and print examples using the trained model.

    Args:
        model_weights_path (str): Path to the trained model weights.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        dataset (Dataset): The dataset containing input prompts.
        num_examples (int, optional): Number of examples to generate. Defaults to 10.
        max_length (int, optional): Maximum length of the generated sequences. Defaults to 512.
    """
    # Load the trained model
    model = AutoModelForCausalLM.from_pretrained(model_weights_path)
    model.to(device)
    model.eval()

    # Ensure num_examples does not exceed the dataset size
    num_examples = min(num_examples, len(dataset))

    for i in tqdm(range(num_examples), desc="Generating examples"):
        # Retrieve the formatted chat from the dataset
        formatted_chat = dataset['formatted_chat'][i]

        # Tokenize the input
        inputs = tokenizer(formatted_chat, return_tensors="pt", truncation=False).to(device)

        # Generate the model's response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=5)

        # Decode the generated response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Print the example
        #print(formatted_chat)
        print(f"Example {i + 1}:\n{generated_text}\n{'=' * 50}\n")


def main():

    args = parse_arguments()

    args.split = "test" if args.use_test else "train"

    print("Loading LLM...")
    llm_id = args.llm_id

    bgm = BGM(
        llm_id, device, quantization_bits=args.quantization_bits,  
        model_max_length=args.model_max_length
    )

    model= bgm.model

    tokenizer = bgm.tokenizer

    model, tokenizer = setup_chat_format(model, tokenizer)
    print("LLM loaded")

    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    print("Corpus and search results loaded")

    task_instruction = args.task_instruction

    print("Loading prompt dataset...")
    dataset_training, dataset_testing = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, tokenizer, task_instruction
    )
    print("Prompt dataset loaded")

    print_info(args)

    print(dataset_training['formatted_chat'][0])
    print(dataset_testing['formatted_chat'][0])

    #bgm_training(args, model, bgm.tokenizer, dataset_training)

    #training_path=r'C:\Users\franc\Documents\Bridge_the_GAP\data\SFT_training_bgm\meta-llama-Llama-3.2-1B\checkpoint-700'  #best checkpoint 700-800
    #evaluate_model(training_path, bgm.tokenizer, dataset_testing, num_examples=20, max_length=15)



if __name__ == "__main__":
    seed_everything(SEED)
    main()