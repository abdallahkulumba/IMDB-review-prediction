import torch.nn as nn
from models.base_model import BaseModel
from typing import Any
import torch

class RNNModel(BaseModel):
    def __init__(self, config: Any, tokenizer: Any):
        super().__init__(config, tokenizer)
        
        self.embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=config.embedding_dim
        )
        
        self.rnn = nn.RNN(
            input_size=config.embedding_dim,
            hidden_size=config.rnn_units,
            batch_first=True,
            nonlinearity='tanh'
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(config.rnn_units, config.dense_units)
        self.fc2 = nn.Linear(config.dense_units, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x) -> torch.Tensor:
        if isinstance(x, dict):  # BERT case handled in base model
            raise ValueError("RNNModel doesn't support BERT inputs")
            
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # RNN
        _, hidden = self.rnn(embedded)  # hidden: (1, batch_size, rnn_units)
        hidden = hidden.squeeze(0)  # (batch_size, rnn_units)
        
        # Fully connected
        x = self.dropout(hidden)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x