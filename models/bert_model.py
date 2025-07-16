from transformers import BertModel
import torch.nn as nn
from models.base_model import BaseModel
from typing import Any
import torch

class BERTModel(BaseModel):
    def __init__(self, config: Any, tokenizer: Any):
        super().__init__(config, tokenizer)
        
        self.bert = BertModel.from_pretrained(config.model_name)
        
        # Freeze BERT layers except the last n
        for param in self.bert.parameters():
            param.requires_grad = False
            
        if config.trainable_layers > 0:
            for layer in self.bert.encoder.layer[-config.trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, dict):
            raise ValueError("BERTModel requires dictionary inputs")
        
        for k,v in x.items():
            if v.dim( )==1:
                raise ValueError(f"{k} must be 2D (batch_size, seq_len), got shape: {v.shape}")
        
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        token_type_ids = x['token_type_ids']     

        # Optional: Add assert to catch shape issues early
        assert input_ids.ndim == 2, f"Expected input_ids to be 2D, got {input_ids.shape}"
        assert attention_mask.ndim == 2, f"Expected attention_mask to be 2D, got {attention_mask.shape}"
        assert token_type_ids.ndim == 2, f"Expected token_type_ids to be 2D, got {token_type_ids.shape}"
   
            
        outputs = self.bert(
            input_ids= input_ids,
            attention_mask= attention_mask,
            token_type_ids= token_type_ids,
            return_dict=True
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits