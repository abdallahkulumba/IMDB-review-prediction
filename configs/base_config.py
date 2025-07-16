from dataclasses import dataclass

@dataclass
class BaseConfig:
    # Dataset parameters
    dataset_name: str = "imdb_reviews"
    max_seq_length: int = 256
    batch_size: int = 16
    test_size: float = 0.2
    random_seed: int = 42
    
    # Training parameters
    epochs: int = 1
    learning_rate: float = 1e-3
    early_stopping_patience: int = 2  # Fixed typo (patiences -> patience)
    
    # Paths (updated structure)
    data_dir: str = "data/raw/aclImdb"
    processed_data_dir: str = "data/processed"
    output_dir: str = "outputs/"
    
    # GloVe parameters
    glove_dir: str = "data/glove"
    glove_file: str = "glove.6B.100d.txt"  # Specific file
    
    # Model/output paths
    model_save_dir: str = "outputs/models/"
    results_dir: str = "outputs/results/"
    
    # Logging
    log_dir: str = "outputs/logs/"
    experiment_name: str = "model_comparison"