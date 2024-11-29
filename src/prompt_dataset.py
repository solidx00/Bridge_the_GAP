import json
import random
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import normalize_text
from normalize_answers import *


class QueryDataset(Dataset):
    """
    A dataset class for managing queries data into structured prompts suitable for input to LLMS.

    Attributes:
        data_path (str): Path to the dataset file containing the query and related information.
        model_name (str): The name of the language model used for generating answers.
        do_normalize_query (bool): Flag to determine if text normalization is applied to the query.
    """
    def __init__(
        self, 
        data_path: str, 
        model_name: str,
        do_normalize_query: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.model_name = model_name
        self.do_normalize_query = do_normalize_query
        self._load_data()


    def _load_data(self):
        """
        Loads data from the specified path and processes it.
        """
        try:
            with open(self.data_path, "r") as fin:
                data = json.load(fin)
            self.process_file_data(data)
        except IOError as e:
            print(f"Error reading file {self.data_path}: {e}")


    def process_file_data(self, data: List[Dict]):
        """ Processes each example in the dataset to prepare prompts for the LLM. """  
        self.questions = []
        self.example_ids = []

        for example in data:
            self.example_ids.append(example['example_id'])

            if 'query' in example:
                question = example['query']
            elif 'question' in example:
                question = example['question']
            else:
                raise ValueError("No 'query' or 'question' key in example")
            
            if self.do_normalize_query:
                question = normalize_text.normalize(question)
            self.questions.append(question)


    def build_qa_prompt(self, query: str) -> str:
        task_instruction = "You are given a question and you must respond based on the provided documents. You must always provide an answer."
        prompt = f"""{task_instruction}\nQuestion: {query}\nAnswer:"""
        
        # Custom prompt format for mpt models
        if 'mpt' in self.model_name:
            INSTRUCTION_KEY = "### Instruction:"
            RESPONSE_KEY = "### Response:"
            INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            PROMPT_FOR_GENERATION_FORMAT = """{intro}\n{instruction_key}\n{instruction}\n{response_key}""".format(
                intro=INTRO_BLURB,
                instruction_key=INSTRUCTION_KEY,
                instruction="{instruction}",
                response_key=RESPONSE_KEY,
            )
            prompt = PROMPT_FOR_GENERATION_FORMAT.format(
                instruction=prompt[:-8]
            )

        return prompt


    def __getitem__(self, idx: int):   
        prompt = self.build_qa_prompt(self.questions[idx])

        return {
            "example_id": self.example_ids[idx],
            "query": self.questions[idx],
            "prompt": prompt,
        }

    def __len__(self):
        return len(self.example_ids)


def hash_document(text: str) -> str:
    """
    Generate a SHA-256 hash for a given text.
    """
    return hashlib.sha256(text.encode()).hexdigest()


class PromptDataset(Dataset):
    """
    A dataset class for managing, preprocessing, and organizing document data into structured prompts suitable for input to LLMS.

    Attributes:
        corpus (List[Dict]): The list containing the document corpus.
        data_path (str): Path to the dataset file containing the query and related information.
        tokenizer (AutoTokenizer): The tokenizer used to tokenize the prompt, in order to check its tokenized length.
        max_tokenized_length (int): The maximum length of tokenized prompt. Prompts that exceed this length are excluded from the dataset.
        search_results (List[Tuple[List[str], List[float]]]): A list of tuples containing document indices and their scores. The results may come from a retriever.
        full_to_subset_idx_map (Dict[int, int]): Dictionary that maps the indices in the full corpus to the given subset (corpus).
        do_normalize_query (bool): Flag to determine if text normalization is applied to the query.
        num_documents_in_context (int): The total number of documents to consider in the context.
        gold_position (int): The specific position (0-indexed) of the gold document in the context.
        randomize_gold_position (bool): Flag to determine if the gold document position should be random.
        get_documents_without_answer (bool): Flag to determine if documents without the answer should be included in the prompt.
    """
    def __init__(
        self, 
        corpus: List[Dict],
        data_path: str,  
        tokenizer: AutoTokenizer,
        max_tokenized_length: int,
        search_results: List[Tuple[List[int], List[float]]],
        full_to_subset_idx_map: Dict[int, int] = None,
        do_normalize_query: bool = False,
        num_documents_in_context: int = 5,
        gold_position: int = None,
        randomize_gold_position: bool = False,
        get_documents_without_answer: bool = False,
    ):
        super().__init__()
        self.corpus = corpus
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_tokenized_length = max_tokenized_length
        self.search_results = search_results
        self.full_to_subset_idx_map = full_to_subset_idx_map
        self.do_normalize_query = do_normalize_query
        self.num_documents_in_context = num_documents_in_context
        self.gold_position = gold_position
        self.randomize_gold_position = randomize_gold_position
        self.get_documents_without_answer = get_documents_without_answer
    
        
        self._validate_initialization_parameters()
        self._load_data()


    def _validate_initialization_parameters(self):
        """Validates initialization parameters for logical consistency and correctness."""
        if self.num_documents_in_context <= 0:
            raise ValueError("num_documents_in_context must be positive.")
        
        if self.max_tokenized_length <= 0:
            raise ValueError("max_tokenized_length must be positive.")

        if self.gold_position is not None:
            if self.gold_position < 0 or (self.gold_position >= self.num_documents_in_context):
                raise ValueError(f"Invalid gold position: {self.gold_position}")
        
        if self.gold_position is not None and self.randomize_gold_position:
            raise ValueError("Both 'gold_position' and 'randomize_gold_position' cannot be set at the same time.")


    def _load_data(self):
        """
        Loads data from the specified path and processes it.
        """
        try:
            with open(self.data_path, "r") as fin:
                data = json.load(fin)
            self.process_file_data(data)
        except IOError as e:
            print(f"Error reading file {self.data_path}: {e}")


    def process_file_data(self, data: List[Dict]):  
        """
        Processes each example in the dataset to prepare prompts for the LLM.

        This involves assembling document contexts, normalizing text as needed,
        and checking against the maximum token length to ensure compatibility with the LLM's input specifications.

        Args:
            data (List[Dict]): The dataset, where each entry contains information about an example,
            including the example's ID, the gold document index, answers, and the query.
        """
        self.example_ids = []
        self.queries = []
        self.prompts = []
        self.gold_document_idxs = []
        self.excluded_samples_ids = []
        self.preprocessed_data = []
        self.prompt_tokens_lengths = []

        for idx, example in enumerate(data):
            example_id = str(example['example_id'])
            gold_document_idx = str(example['idx_gold_in_corpus'])
            answers = example['answers']

            formatted_documents, document_indices = self.prepare_documents_for_prompt(
                idx, gold_document_idx, answers
            )

            # Build the prompt
            documents_str = '\n'.join(formatted_documents)
            query = example['question']
            if self.do_normalize_query:
                query = normalize_text.normalize(query)
            prompt = self.build_qa_prompt(query, documents_str)

            # Check if the prompt exceeds 'max_tokenized_length'
            tokens = self.tokenizer.tokenize(prompt)
            tokens_len = len(tokens)
            if tokens_len >= self.max_tokenized_length:
                self.excluded_samples_ids.append((idx, example_id))
                print("Skipping example {} due to prompt length.".format((idx, example_id)))
                continue  # Skip adding this example

            if len(formatted_documents) != self.num_documents_in_context:
                print(f"Warning: Not enough documents for example {idx}.")

            # If prompt is within limit, add to preprocessed data
            self.preprocessed_data.append((formatted_documents, list(document_indices)))
            self.example_ids.append(example_id)
            self.queries.append(query)
            self.prompts.append(prompt)
            self.gold_document_idxs.append(gold_document_idx)
            self.prompt_tokens_lengths.append(tokens_len)


    def prepare_documents_for_prompt(
        self, 
        example_idx: int, 
        gold_document_idx: int, 
        answers: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Prepares and formats a set of documents for inclusion in a prompt, including the insertion of a gold document at the appropriate position.

        This function performs several key steps to prepare documents for a prompt:
        1. Retrieves document indices based on the example index.
        2. Inserts the gold document index into the retrieved list of indices at a specified or randomized position, if necessary.
        3. Formats the documents corresponding to the updated list of indices, preparing them for inclusion in the prompt. 
           This includes potentially filtering documents based on answers or other criteria.

        Args:
            example_idx (int): The index of the current example in the dataset. This is used to retrieve the appropriate set of document indices.
            gold_document_idx (int): The index of the gold document within the corpus. 
            answers (List[str]): A list of answers that can be used to ensure the relevance of documents included in the prompt.

        Returns:
            A tuple containing two lists:
            - The first list contains the formatted documents.
            - The second list contains the indices of the included documents.
        """
        indices = self._get_indices(example_idx)
        updated_indices, gold_position = self._insert_gold_document_idx(
            indices, gold_document_idx
        )

        # Get the documents and their indices in the corpus
        formatted_documents, document_indices = self._get_documents(
            updated_indices, answers, gold_document_idx, gold_position
        )
        return formatted_documents, document_indices


    def _get_indices(self, example_idx: int) -> List[int]:
        """ Get the indices in the corpus of the documents retrieved possibly by a retriever. """
        indices, scores = self.search_results[example_idx]
        return indices


    def _insert_gold_document_idx(
        self, 
        indices: List[int], 
        gold_document_idx: int
    ) -> Tuple[List[int], int]:
        """
        Inserts the index of a gold document into the provided list of indices at a specified or random position.

        Args:
            indices: A list of integers representing document indices.
            gold_document_idx: The index of the gold document to insert.

        Returns:
            A tuple containing:
            - The updated list of indices with the gold document index inserted.
            - The position at which the gold document index was inserted.
        """
        gold_position = None
        
        if self.gold_position is not None:
            # Direct insertion
            gold_position = self.gold_position
            indices = indices[:gold_position] + [gold_document_idx] + indices[gold_position:]
        elif self.randomize_gold_position:
            # Insert at a random position
            gold_position = random.randint(0, self.num_documents_in_context - 1)
            indices = indices[:gold_position] + [gold_document_idx] + indices[gold_position:]
        return indices, gold_position


    def _get_documents(    
        self,
        indices: List[int],
        answers: List[str],
        gold_document_idx: Optional[int],
        gold_position: Optional[int]
    ) -> Tuple[List[str], List[int]]:
        """ Choose the appropriate method based on the flag """
        if self.get_documents_without_answer:
            return self._get_answerless_documents_from_indices(
                indices, answers, gold_document_idx, gold_position
            )
        else:
            return self._get_documents_from_indices(indices)
            

    def _get_documents_from_indices(self, indices: List[int]) -> Tuple[List[str], List[int]]:
        """
        Selects documents from the corpus based on provided indices and formats them.
        Handles both full corpus and subsets by mapping indices if necessary.

        Args:
            indices: A list of integers representing the positions of documents to retrieve in the corpus.

        Returns:
            A tuple containing two lists:
            - The first list contains the formatted documents.
            - The second list contains the indices of the included documents.
        """
        formatted_documents = []
        
        # Full corpus
        if self.full_to_subset_idx_map is None:
            documents_info = [self.corpus[i] for i in map(int, indices)]
        else: 
            documents_info: List[Dict] = []
            # 'indices' are from the full corpus, so we need to map them to the subset
            for i in map(int, indices):
                documents_info.append(self.corpus[self.full_to_subset_idx_map[i]])
        
        seen_hashes = set()
        # List to store the indices of documents actually added
        document_indices = []  
        for doc_info in documents_info:
            if len(formatted_documents) == self.num_documents_in_context:
                break
            
            doc_idx = doc_info['full_corpus_idx']
            title = doc_info['title']
            text = doc_info['text']

            doc_hash = hash_document(text)
            # Skip the document if it is a duplicate
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)
            
            doc_str = f"Document [{doc_idx}](Title: {title}) {text}"
            formatted_documents.append(doc_str)
            document_indices.append(doc_idx)

        return formatted_documents, document_indices
    

    def _get_answerless_documents_from_indices(
        self,
        indices: List[int],
        answers: List[str],
        gold_document_idx: Optional[int],
        gold_position: Optional[int]
    ) -> Tuple[List[str], List[int]]:
        """
        Selects documents from the corpus that do not contain any of the given answers, optionally including
        a specific 'gold' document at a designated position.

        Args:
            indices: A list of integers representing the indices of documents to retrieve from the corpus.
            answers: A list of strings representing the answers to exclude from the documents.
            gold_document_idx: The index of the gold document in the full corpus.
            gold_position: The desired position of the gold document within the returned list, if any.

        Returns:
            A tuple containing two lists:
            - The first list contains the documents that do not contain the answer and possibly the gold.
            - The second list contains the indices of the included documents.
        """
        # Full corpus
        if self.full_to_subset_idx_map is None:
            documents_info = [self.corpus[i] for i in map(int, indices)]
        else: 
            documents_info: List[Dict] = []
            # 'indices' are from the full corpus, so we need to map them to the subset
            for i in map(int, indices):
                documents_info.append(self.corpus[self.full_to_subset_idx_map[i]])

        answerless_documents = []
        gold_document = None
        seen_hashes = set()
        # List to store the indices of documents actually added
        document_indices = [] 

        for doc_info in documents_info:
            doc_idx = doc_info['full_corpus_idx']
            title = doc_info['title']
            text = doc_info['text']

            doc_hash = hash_document(text)
            # Skip the document if it's a duplicate
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)

            if str(doc_idx) == gold_document_idx:
                gold_document = f"Document [{doc_idx}](Title: {title}) {text}"
                continue
            
            if not is_answer_in_text(text, answers):
                answerless_doc = f"Document [{doc_idx}](Title: {title}) {text}"
                answerless_documents.append(answerless_doc)
                document_indices.append(doc_idx)

        # Insert gold document at the specified/random position
        if gold_position is not None and gold_document is not None:
            gold_position = min(gold_position, len(answerless_documents))
            answerless_documents.insert(gold_position, gold_document)
            document_indices.insert(gold_position, gold_document_idx)

        # Limit the number of documents to the specified context size
        docs = answerless_documents[:self.num_documents_in_context]
        indices = document_indices[:self.num_documents_in_context]
        return docs, indices



    def build_qa_prompt(self, query: str, documents_str: str) -> str:
        task_instruction = "You are given a question and you must respond based on the provided documents. You must always provide an answer."
        prompt = f"""{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query}\nAnswer:"""

        # Custom prompt format for mpt models
        if 'mpt' in self.tokenizer.name_or_path:
            INSTRUCTION_KEY = "### Instruction:"
            RESPONSE_KEY = "### Response:"
            INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            PROMPT_FOR_GENERATION_FORMAT = """{intro}\n{instruction_key}\n{instruction}\n{response_key}""".format(
                intro=INTRO_BLURB,
                instruction_key=INSTRUCTION_KEY,
                instruction="{instruction}",
                response_key=RESPONSE_KEY,
            )
            prompt = PROMPT_FOR_GENERATION_FORMAT.format(
                instruction=prompt[:-8]
            )

        return prompt


    def __getitem__(self, idx: int):
        _, document_indices = self.preprocessed_data[idx]

        return {
            "example_id": self.example_ids[idx],
            "query": self.queries[idx],
            "prompt": self.prompts[idx],
            "document_indices": document_indices,
            "gold_document_idx": self.gold_document_idxs[idx],
            "prompt_tokens_len": self.prompt_tokens_lengths[idx]
        }
    

    def __len__(self):
        return len(self.example_ids)