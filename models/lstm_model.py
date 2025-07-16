import torch.nn as nn
from models.base_model import BaseModel
from typing import Any
import torch

class LSTMModel(BaseModel):
    def __init__(self, config: Any, tokenizer: Any):
        super().__init__(config, tokenizer)
        
        self.embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=config.embedding_dim
        )
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.lstm_units,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        
        lstm_output_size = config.lstm_units * 2 if config.bidirectional else config.lstm_units
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(lstm_output_size, config.dense_units)
        self.fc2 = nn.Linear(config.dense_units, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x) -> torch.Tensor:
        if isinstance(x, dict):
            raise ValueError("LSTMModel doesn't support BERT inputs")
            
        embedded = self.embedding(x)
        
        # LSTM
        _, (hidden, _) = self.lstm(embedded)
        
        # Handle bidirectional case
        if self.config.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            hidden = hidden[-1]
            
        # Fully connected
        x = self.dropout(hidden)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x