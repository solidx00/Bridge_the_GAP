import json
import random
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

import json
import random
from torch.utils.data import DataLoader, Dataset

import json
import random
from torch.utils.data import Dataset

def hash_document(text: str) -> str:
    """
    Generate a SHA-256 hash for a given text.
    """
    return hashlib.sha256(text.encode()).hexdigest()

class BGMPromptDataset(Dataset):
    """
    A dataset class for managing, preprocessing, and organizing document data into structured prompts suitable for input to LLMS.

    Attributes:
        corpus (List[Dict]): The list containing the document corpus.
        input_file (str): Path to the dataset file containing the query and related information.
        full_to_subset_idx_map (Dict[int, int]): Dictionary that maps the indices in the full corpus to the given subset (corpus).
        task_instruction (str): The task instruction to add in the prompt for instruct LLM.
    """

    def __init__(
        self, 
        corpus: List[Dict],
        data_path: str, 
        max_tokenized_length: int,
        tokenizer: AutoTokenizer,
        #task_instruction: str,
        full_to_subset_idx_map: Dict[int, int] = None,
    ):
        super().__init__()
        self.id_mapping = {}  # Mappatura ID originale -> ID personalizzato
        self.reverse_id_mapping = {}  # Mappatura inversa ID personalizzato -> ID originale
        self.data_path = data_path
        self.max_tokenized_length = max_tokenized_length
        self.full_to_subset_idx_map = full_to_subset_idx_map
        self.tokenizer = tokenizer
        #self.task_instruction = task_instruction
        self.corpus = corpus
        self.dataset = []
        self.case_counters = {}
        self.process_data()

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        return examples

    def filter_valid_examples(self, examples):
        return [ex for ex in examples if ex["are_answer"] is True]

    
    def check_lenghts_model(self, prompt):
        # Check if the prompt exceeds 'max_tokenized_length'
        tokens = self.tokenizer.tokenize(prompt)
        tokens_len = len(tokens)
        if tokens_len >= self.max_tokenized_length:

            return False
          
        return True

    def process_examples(self, valid_examples):
        """
        Processes all valid examples (where 'are_answer' is True), ensuring that:
        - All examples are included.
        - 'selected_documents' is shuffled if it contains more than one document.
        - The prompt does not exceed the model's token limit.
        - Keeps track of the number of examples based on the number of selected documents.
        - Handles cases where there are zero selected documents.
    
        Args:
            valid_examples (list): List of valid examples (filtered with are_answer=True).
    
        Returns:
            None. The processed examples are appended to `self.dataset`.
        """
    
        # Dictionary to count examples by document count
        doc_count_distribution = {}

        for example in valid_examples:
            prompt = f"Question: {example['query']}"
            retrieved_docs = example["document_indices"]
            selected_docs = example["selected_documents"]

            # Format documents for the prompt
            formatted_documents = self.prepare_documents_for_prompt(retrieved_docs)
            document_str = '\n'.join(formatted_documents)
            full_prompt = f"{prompt}\nDocuments:{document_str}"

            # Check if prompt length is within model constraints
            if self.check_lenghts_model(full_prompt):
                # If there are no selected documents, append an empty list
                if len(selected_docs) == 0:
                    self.dataset.append(self.create_entry(full_prompt, [], []))
                else:
                    self.dataset.append(self.create_entry(full_prompt, retrieved_docs, selected_docs))
            else:
                print(f"Skipping example {example['example_id']} due to prompt length.")
                continue

            # Track document count distribution
            doc_count = len(selected_docs)
            doc_count_distribution[doc_count] = doc_count_distribution.get(doc_count, 0) + 1

        print(f"Total processed examples: {len(self.dataset)}")
        print("Distribution of examples by document count:")
        for count, quantity in sorted(doc_count_distribution.items()):
            print(f"{count} documents: {quantity} examples")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_dataloader(self):
        return self.dataset

    def process_data(self):
        examples = self.load_data()
        valid_examples = self.filter_valid_examples(examples)

        self.process_examples(valid_examples)

    def generate_custom_id(self, original_id: int, idx: int) -> str:
        """
        Genera un ID personalizzato per un documento dato il suo ID originale.
        """
        custom_id = f"Id_{idx + 1}"
        self.id_mapping[original_id] = custom_id
        self.reverse_id_mapping[custom_id] = original_id
        return custom_id
    
    
    def prepare_documents_for_prompt(self, doc_indices: List[int]) -> List[str]:
        """
        Prepara e formatta i documenti per il prompt, utilizzando ID personalizzati.
        """
        formatted_documents = []
        for idx, original_id in enumerate(doc_indices):
            custom_id = self.generate_custom_id(original_id, idx)
            doc_info = self.corpus[self.full_to_subset_idx_map[original_id]]
            title = doc_info['title']
            text = doc_info['text']

            doc_str = f"Document [{custom_id}](Title: {title}) {text}"
            formatted_documents.append(doc_str)
        return formatted_documents

    def create_entry(self, prompt: str, retrieved_docs: List[int], selected_docs: List[int]) -> Dict:
        """
        Crea un esempio, includendo il dizionario di mappatura tra ID personalizzati e normali.
        Mantiene gli ID normali nell'output.
        """
        id_mapping = {}
        mapped_output = []  # Se vogliamo far tornare gli indici mappati [Id_1,Id_2]

        for idx, original_id in enumerate(retrieved_docs):
            custom_id = f"Id_{idx + 1}"
            id_mapping[custom_id] = original_id
            self.id_mapping[original_id] = custom_id
            self.reverse_id_mapping[custom_id] = original_id

            # Se vogliamo far tornare gli indici mappati [Id_1,Id_2]
            if original_id in selected_docs:
                mapped_output.append(custom_id)

        # Se selected_docs è vuoto, restituisci <NO_DOCS>, altrimenti concatenazione degli ID mappati
        #mapped_output_str = "NO_DOCS" if not mapped_output else ", ".join(mapped_output)
        
        if not mapped_output:
            mapped_output_str = "NO_DOCS"
        else:
            # Shuffle se ci sono più documenti
            if len(mapped_output) > 1:
                random.shuffle(mapped_output)
            mapped_output_str = ", ".join(mapped_output)

        entry = {
            "input": {
                "prompt": prompt,
                "retrieved_docs": id_mapping,  # Dizionario {Id_1: id normale}
            },
            "output": mapped_output_str,
        }
        return entry
    
    def remap_output(self, mapped_ids: List[str]) -> List[int]:
        """
        Rimappa gli ID personalizzati agli ID originali.
        """
        return [self.reverse_id_mapping[mapped_id] for mapped_id in mapped_ids]