import torch
import re
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
)
from typing import List, Tuple, Dict

class BGM:
    """
    Bridge Model for selecting and ranking documents by generating document IDs.
    """
    def __init__(
        self, 
        model_id: str, 
        device: str = 'cuda', 
        model_max_length: int = 512
    ):
        self.device = device
        self.model_max_length = model_max_length

        # Initialize the seq2seq model and tokenizer
        self.model, self.tokenizer = self._initialize_model_tokenizer(model_id)

    def _initialize_model_tokenizer(self, model_id: str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        Initializes the seq2seq model and tokenizer with the given model ID.
        """
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_config.max_seq_len = self.model_max_length

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        model.eval()  # Set the model to evaluation mode

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            model_max_length=self.model_max_length,
            padding_side="left",
            truncation_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token

        return model, tokenizer

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 50
    ) -> List[str]:
        """
        Generates the ordered document IDs based on the query and documents.
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=self.model_max_length, 
            padding=True, 
            truncation=True
        ).to(self.device)

        # Generate output
        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,  # Deterministic output
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Decode output
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]