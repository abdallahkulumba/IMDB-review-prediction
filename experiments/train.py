# experiments/train.py

import os
import pickle
import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers  import TensorBoardLogger
from torch.utils.data          import DataLoader

from configs.base_config        import BaseConfig
from configs.model_configs      import (
    RNNConfig,
    LSTMConfig,
    GRUConfig,
    GloVeConfig,
    BERTConfig,
    TransformerConfig,
)
from models.rnn_model           import RNNModel
from models.lstm_model          import LSTMModel
from models.gru_model           import GRUModel
from models.glove_model         import GloVeModel
from models.bert_model          import BERTModel
from models.transformer_model   import TransformerModel

from utils.data_loader          import create_data_loaders
from utils.tokenizer            import PyTorchTokenizer, BertTokenizer
from utils.preprocessor         import TextPreprocessor


def load_data(config: BaseConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess IMDB data."""
    preprocessor = TextPreprocessor()
    df = preprocessor.load_imdb_data(config.data_dir)
    df['text'] = preprocessor.clean_batch(df['text'].tolist())
    train_df = df.sample(frac=0.8, random_state=config.random_seed)
    test_df  = df.drop(train_df.index)
    return train_df, test_df


def train_model(
    model_class: type,
    config: BaseConfig,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    tokenizer
) -> dict:
    """Train and evaluate a single model, return metrics + checkpoint path."""
    model = model_class(config, tokenizer)

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.early_stopping_patience,
        mode='min'
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(config.model_save_dir, 'checkpoints'),
        filename=f"{model_class.__name__}-best",
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=config.experiment_name,
        version=model_class.__name__
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        callbacks=[early_stop, checkpoint],
        logger=logger,
        accelerator='auto',
        devices='auto',
        enable_progress_bar=True
    )

    trainer.fit(model, train_loader, val_loader)
    best_val_loss   = checkpoint.best_model_score
    if hasattr(best_val_loss, 'item'):
        best_val_loss = best_val_loss.item()
    best_model_path = checkpoint.best_model_path

    return {
        'model':     model_class.__name__,
        'val_loss':  best_val_loss,
        'model_path': best_model_path
    }


def train_all_models() -> pd.DataFrame:
    """Train each model in sequence and collect their results."""
    config   = BaseConfig()
    train_df, test_df = load_data(config)
    results  = []

    model_map = {
        'RNN':         (RNNModel,       RNNConfig),
        'LSTM':        (LSTMModel,      LSTMConfig),
        'GRU':         (GRUModel,       GRUConfig),
        'GloVe':       (GloVeModel,     GloVeConfig),
        'BERT':        (BERTModel,      BERTConfig),
        'Transformer': (TransformerModel, TransformerConfig),
    }

    for name, (ModelCls, ConfigCls) in model_map.items():
        print(f"\nTraining {name} model...")

        # Instantiate config & tokenizer
        model_cfg = ConfigCls()
        if name == 'BERT':
            tokenizer = BertTokenizer(model_cfg)
        else:
            tokenizer = PyTorchTokenizer(model_cfg)
            tokenizer.fit(train_df['text'])

            # --- PICKLE TOKENIZER FOR LATER INFERENCE ---
            tok_dir = os.path.join(model_cfg.model_save_dir, 'tokenizers')
            os.makedirs(tok_dir, exist_ok=True)
            tok_path = os.path.join(tok_dir, f"{name}_tokenizer.pkl")
            with open(tok_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            # ---------------------------------------------

        # Build DataLoaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df=train_df,
            test_df=test_df,
            tokenizer=tokenizer,
            batch_size=model_cfg.batch_size,
            val_ratio=0.2
        )

        # Train & record metrics
        metrics = train_model(
            ModelCls,
            model_cfg,
            train_loader,
            val_loader,
            tokenizer
        )
        results.append(metrics)

    # Save comparison table
    results_df = pd.DataFrame(results)
    os.makedirs(config.results_dir, exist_ok=True)
    results_df.to_csv(
        os.path.join(config.results_dir, 'model_comparison.csv'),
        index=False
    )
    return results_df


if __name__ == "__main__":
    df = train_all_models()
    print("\nTraining complete. Results:")
    print(df[['model', 'val_loss', 'model_path']])
