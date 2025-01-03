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
        task_instruction: str,
        full_to_subset_idx_map: Dict[int, int] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_tokenized_length = max_tokenized_length
        self.full_to_subset_idx_map = full_to_subset_idx_map
        self.tokenizer = tokenizer
        self.task_instruction = task_instruction
        self.corpus = corpus
        self.percentages = {
            "case_1_single_doc": 0.07,
            "case_2_multiple_docs": 0.4,
            "case_3_no_docs": 0.1,
            "case_4_multi_doc_unchanged": 0.35,
            "case_5_reranking": 0.5,
            "case_6_single_doc_unchanged": 0.05,
        }
        self.dataset = []
        self.case_counters = {}
        self.process_data()

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        return examples

    def filter_valid_examples(self, examples):
        return [ex for ex in examples if ex["are_answer"] is True]

    def group_examples(self, valid_examples):
        return {
            "len_0": [ex for ex in valid_examples if len(ex["selected_documents"]) == 0],
            "len_1": [ex for ex in valid_examples if len(ex["selected_documents"]) == 1],
            "len_gt_1": [ex for ex in valid_examples if len(ex["selected_documents"]) > 1],
        }

    def calculate_case_limits(self, grouped_examples):
        return {
            "case_1_single_doc": int(len(grouped_examples["len_1"]) * self.percentages["case_1_single_doc"]),
            "case_2_multiple_docs": int(len(grouped_examples["len_gt_1"]) * self.percentages["case_2_multiple_docs"]),
            "case_3_no_docs": int(len(grouped_examples["len_0"]) * self.percentages["case_3_no_docs"]),
            "case_4_multi_doc_unchanged": int(len(grouped_examples["len_gt_1"]) * self.percentages["case_4_multi_doc_unchanged"]),
            "case_5_reranking": int(len(grouped_examples["len_gt_1"]) * self.percentages["case_5_reranking"]),
            "case_6_single_doc_unchanged": int(len(grouped_examples["len_1"]) * self.percentages["case_6_single_doc_unchanged"]),
        }
    
    def check_lenghts_model(self, prompt):
        # Check if the prompt exceeds 'max_tokenized_length'
        tokens = self.tokenizer.tokenize(prompt)
        tokens_len = len(tokens)
        if tokens_len >= self.max_tokenized_length:

            return False
          
        return True

    def process_examples(self, valid_examples, group_case_limits):
        self.case_counters = {case: 0 for case in group_case_limits}


        for example in valid_examples:
            prompt = f"Task Instruction: {self.task_instruction}\nQuestion:{example['query']}"
            retrieved_docs = example["document_indices"]
            selected_docs = example["selected_documents"]

            if len(selected_docs) == 1 and self.case_counters["case_1_single_doc"] < group_case_limits["case_1_single_doc"]:
                formatted_documents = self.prepare_documents_for_prompt(retrieved_docs)
                #Build the prompt
                document_str = '\n'.join(formatted_documents)
                prompt += f'\nDocuments:{document_str}'

                if self.check_lenghts_model(prompt) == True:
                    self.dataset.append(self.create_entry(prompt, retrieved_docs, selected_docs))
                else:
                    print("Skipping example {} due to prompt length.".format(example))
                    continue
                #self.dataset.append(self.create_entry(prompt, retrieved_docs, selected_docs))
                self.case_counters["case_1_single_doc"] += 1

            elif len(selected_docs) > 1 and self.case_counters["case_2_multiple_docs"] < group_case_limits["case_2_multiple_docs"]:
                formatted_documents = self.prepare_documents_for_prompt(retrieved_docs)
                #Build the prompt
                document_str = '\n'.join(formatted_documents)
                prompt += f'\nDocuments:{document_str}'

                if self.check_lenghts_model(prompt) == True:
                    self.dataset.append(self.create_entry(prompt, retrieved_docs, selected_docs))
                else:
                    print("Skipping example {} due to prompt length.".format(example))
                    continue

                #self.dataset.append(self.create_entry(prompt, retrieved_docs, selected_docs))
                self.case_counters["case_2_multiple_docs"] += 1

            elif len(selected_docs) == 0 and self.case_counters["case_3_no_docs"] < group_case_limits["case_3_no_docs"]:
                self.dataset.append(self.create_entry(prompt, [], []))
                self.case_counters["case_3_no_docs"] += 1

            elif len(selected_docs) > 1 and self.case_counters["case_4_multi_doc_unchanged"] < group_case_limits["case_4_multi_doc_unchanged"]:
                formatted_documents = self.prepare_documents_for_prompt(selected_docs)
                #Build the prompt
                document_str = '\n'.join(formatted_documents)
                prompt += f'\nDocuments:{document_str}'

                if self.check_lenghts_model(prompt) == True:
                    self.dataset.append(self.create_entry(prompt, selected_docs, selected_docs))
                else:
                    print("Skipping example {} due to prompt length.".format(example))
                    continue

                #self.dataset.append(self.create_entry(prompt, selected_docs, selected_docs))
                self.case_counters["case_4_multi_doc_unchanged"] += 1

            elif len(selected_docs) == 1 and self.case_counters["case_6_single_doc_unchanged"] < group_case_limits["case_6_single_doc_unchanged"]:
                formatted_documents = self.prepare_documents_for_prompt(selected_docs)
                #Build the prompt
                document_str = '\n'.join(formatted_documents)
                prompt += f'\nDocuments:{document_str}'

                if self.check_lenghts_model(prompt) == True:
                    self.dataset.append(self.create_entry(prompt, selected_docs, selected_docs))
                else:
                    print("Skipping example {} due to prompt length.".format(example))
                    continue

                #self.dataset.append(self.create_entry(prompt, selected_docs, selected_docs))
                self.case_counters["case_6_single_doc_unchanged"] += 1

            elif len(selected_docs) > 1 and self.case_counters["case_5_reranking"] < group_case_limits["case_5_reranking"]:
                reranked_docs = selected_docs[:]
                while reranked_docs == selected_docs:
                    reranked_docs = random.sample(selected_docs, len(selected_docs))

                formatted_documents = self.prepare_documents_for_prompt(reranked_docs)
                #Build the prompt
                document_str = '\n'.join(formatted_documents)
                prompt += f'\nDocuments:{document_str}'

                if self.check_lenghts_model(prompt) == True:
                    self.dataset.append(self.create_entry(prompt, reranked_docs, selected_docs))
                else:
                    print("Skipping example {} due to prompt length.".format(example))
                    continue

                #self.dataset.append(self.create_entry(prompt, reranked_docs, selected_docs))
                self.case_counters["case_5_reranking"] += 1

    def create_entry(self, prompt, retrieved_docs, selected_docs):

        entry = {
            "input": {
                "prompt": prompt,
                "retrieved_docs": retrieved_docs,
            },
            "output": selected_docs,
        }
        return entry

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_dataloader(self):
        return self.dataset

    def process_data(self):
        examples = self.load_data()
        valid_examples = self.filter_valid_examples(examples)

        print(f"Totale esempi nel file di input: {len(examples)}")
        print(f"Esempi con 'are_answer=True': {len(valid_examples)}")

        grouped_examples = self.group_examples(valid_examples)
        for key, group in grouped_examples.items():
            print(f"Esempi con '{key}': {len(group)}")

        group_case_limits = self.calculate_case_limits(grouped_examples)
        print("Distribuzione pianificata degli esempi nel dataset creato:")
        for case, limit in group_case_limits.items():
            print(f"{case}: {limit}")

        self.process_examples(valid_examples, group_case_limits)

        print("Esempi effettivamente inclusi nel dataset creato:")
        tot = 0
        for case, count in self.case_counters.items():
            tot += count
            print(f"{case}: {count}")

        print(f"Totale degli Esempi inclusi nel training dataset creato: {tot}")

    def prepare_documents_for_prompt(self, doc_indices: List[int]) -> Tuple[List[str], List[int]]:

        """
        Prepare and format a set of documents for inclusion in a prompt, including the insertion of a gold document at the appropriate position.

        This function performs several key steps to prepare documents for a prompt:
        1. Retrieves document indices based on the example index.
        2. Formats the documents corresponding to the updated list of indices, preparing them for inclusion in the prompt. 
           This includes potentially filtering documents based on answers or other criteria.

        Args:
            example_idx (int): The index of the current example in the dataset. This is used to retrieve the appropriate set of document indices.

        Returns:
            A tuple containing two lists:
            - List contains the formatted documents.
        """
        indices = doc_indices

        # Get the documents and their indices in the corpus
        formatted_documents = self._get_documents_from_indices(
            indices
        )
        return formatted_documents

    def _get_documents_from_indices(self, indices: List[int]) -> List[str]:
        """
        Selects documents from the corpus based on provided indices and formats them.
        Handles both full corpus and subsets by mapping indices if necessary.

        Args:
            indices: A list of integers representing the positions of documents to retrieve in the corpus.

        Returns:
            - List contains the formatted documents.
        """
        formatted_documents = []

        documents_info: List[Dict] = []
        # 'indices' are from the full corpus, so we need to map them to the subset
        for i in map(int, indices):
            documents_info.append(self.corpus[self.full_to_subset_idx_map[i]])

        seen_hashes = set()
        # List to store the indices of documents actually added
        for doc_info in documents_info:
            if len(formatted_documents) == 5:
                break
            
            doc_idx = doc_info['full_corpus_idx']
            title = doc_info['title']
            text = doc_info['text']

            doc_hash = hash_document(title + " " + text if title != "" else text)
            # Skip the document if it is a duplicate
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)
            
            doc_str = f"Document [{doc_idx}](Title: {title}) {text}"
            formatted_documents.append(doc_str)

        return formatted_documents