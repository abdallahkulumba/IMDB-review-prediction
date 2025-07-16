import torch.nn as nn
import numpy as np
import os
from models.base_model import BaseModel
from typing import Any, Dict
import torch

class GloVeModel(BaseModel):
    def __init__(self, config: Any, tokenizer: Any):
        super().__init__(config, tokenizer)
        
        # Load GloVe embeddings
        self.embedding_matrix = self._create_embedding_matrix()
        
        self.embedding = nn.Embedding(
            num_embeddings=self.embedding_matrix.shape[0],
            embedding_dim=self.embedding_matrix.shape[1],
            _weight=torch.tensor(self.embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.lstm_units,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(config.lstm_units, config.dense_units)
        self.fc2 = nn.Linear(config.dense_units, 1)
        self.relu = nn.ReLU()
        
    def _load_glove_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings_index = {}
        glove_path = os.path.join(self.config.glove_dir, self.config.glove_file)
        
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                
        return embeddings_index
        
    def _create_embedding_matrix(self) -> np.ndarray:
        embeddings_index = self._load_glove_embeddings()
        word_index = self.tokenizer.vocab.get_stoi()
        
        embedding_matrix = np.zeros((len(word_index) + 1, self.config.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
        return embedding_matrix
        
    def forward(self, x) -> torch.Tensor:
        if isinstance(x, dict):
            raise ValueError("GloVeModel doesn't support BERT inputs")
            
        embedded = self.embedding(x)
        
        # LSTM
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]
        
        # Fully connected
        x = self.dropout(hidden)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x