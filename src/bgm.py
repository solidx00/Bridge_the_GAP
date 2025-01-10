import torch
import re
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM,
    BitsAndBytesConfig, LlamaConfig, AutoModelForCausalLM
)
from peft import PeftModel
from typing import List, Tuple, Optional

class BGM:
    """
    Bridge Model for selecting and ranking documents by generating document IDs.
    """
    def __init__(
        self, 
        model_id: str, 
        device: str = 'cuda', 
        quantization_bits: Optional[int] = None,
        model_max_length: int = 4096,
        lora_weights_path: Optional[str] = None  # Path to LoRA weights
    ):
        self.device = device
        self.model_max_length = model_max_length
        self_model_id = model_id
        self.lora_weights_path = lora_weights_path

        self.bnb_config = self._set_quantization(quantization_bits)
        self.model, self.tokenizer = self._initialize_model_tokenizer(model_id)

        # Load LoRA weights if provided
        if self.lora_weights_path:
            self._load_lora_weights()

    def _set_quantization(self, quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
        """
        Configure quantization settings based on the specified number of bits.
        """
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = 'nf4'
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            elif quantization_bits == 8:
                bnb_config.load_in_8bit = True
            return bnb_config
        return None

    def _initialize_model_tokenizer(self, model_id: str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        Initializes the seq2seq model and tokenizer with the given model ID.
        """
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_config.max_seq_len = self.model_max_length

        # Determine the appropriate model class
        if isinstance(model_config, LlamaConfig):
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                config=model_config,
                torch_dtype=torch.bfloat16,
                device_map='cuda:0',
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                config=model_config,
                torch_dtype=torch.bfloat16,
                device_map='cuda:0',
            )
    
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            model_max_length=self.model_max_length,
            padding_side="left",
            truncation_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token

        return model, tokenizer
    
    def _load_lora_weights(self):
        """
        Load LoRA weights into the model.
        """
        try:
            self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path)
            print(f"LoRA weights loaded from: {self.lora_weights_path}")
        except Exception as e:
            raise ValueError(f"Failed to load LoRA weights: {str(e)}")

    def generate(
        self, 
        prompt: str, 
        padding_strategy: str = "longest",
        max_new_tokens: int = 15
    ) -> List[str]:
        """
        Generates the ordered document IDs based on the query and documents.
        """
        if not prompt.strip():  # Handle empty prompts
            return []  # Return an empty list as the output
    
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.model_max_length,
            truncation=True,
            padding=padding_strategy
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]