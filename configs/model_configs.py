from dataclasses import dataclass
from configs.base_config import BaseConfig

@dataclass
class RNNConfig(BaseConfig):
    embedding_dim: int = 128
    rnn_units: int = 64
    dense_units: int = 32
    dropout_rate: float = 0.2

@dataclass
class LSTMConfig(RNNConfig):
    lstm_units: int = 64
    bidirectional: bool = True

@dataclass
class GRUConfig(RNNConfig):
    gru_units: int = 64

@dataclass
class GloVeConfig(BaseConfig):
    embedding_dim: int = 100  # Must match GloVe dimension
    lstm_units: int = 64
    dense_units: int = 32
    glove_file: str = "glove.6B.100d.txt"
    dropout_rate: float = 0.2  # Dropout rate for GloVe model

@dataclass
class BERTConfig(BaseConfig):
    model_name: str = "bert-base-uncased"
    learning_rate: float = 2e-5
    trainable_layers: int = 1  # How many BERT layers to fine-tune

@dataclass
class TransformerConfig(BaseConfig):
    num_layers: int = 2
    d_model: int = 128
    num_heads: int = 4
    dff: int = 512
    dropout_rate: float = 0.1
    #vocab_size: int = 20000
    vocab_size:   int     = None   
    max_seq_length:int     = 128 

