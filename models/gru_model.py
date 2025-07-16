import torch.nn as nn
from models.base_model import BaseModel
from typing import Any
import torch

class GRUModel(BaseModel):
    def __init__(self, config: Any, tokenizer: Any):
        super().__init__(config, tokenizer)
        
        self.embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=config.embedding_dim
        )
        
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.gru_units,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(config.gru_units, config.dense_units)
        self.fc2 = nn.Linear(config.dense_units, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x) -> torch.Tensor:
        if isinstance(x, dict):
            raise ValueError("GRUModel doesn't support BERT inputs")
            
        embedded = self.embedding(x)
        
        # GRU
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)
        
        # Fully connected
        x = self.dropout(hidden)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x