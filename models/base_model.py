import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torchmetrics import Accuracy
from datetime import datetime
import os
import pickle
from utils.tokenizer import BaseTokenizer, BertTokenizer  # Assuming BaseTokenizer is defined in utils.tokenizer

class BaseModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        
    def forward(self, x):
        raise NotImplementedError
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(), y.float())
        self.train_acc(logits.squeeze().sigmoid(), y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(), y.float())
        self.val_acc(logits.squeeze().sigmoid(), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(), y.float())
        self.test_acc(logits.squeeze().sigmoid(), y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
        
    def predict(self, text):
        self.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode([text])
            if isinstance(encoded, dict):  # For BERT
                encoded = {k: torch.tensor(v).to(self.device) for k, v in encoded.items()}
                logits = self(encoded)
            else:
                logits = self(torch.tensor(encoded).to(self.device))
            return torch.sigmoid(logits.squeeze()).item()
            
    def save(self, model_dir=None):
        if model_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_dir = os.path.join(self.config.output_dir, "models", f"{self.__class__.__name__}_{timestamp}")
            
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))
        
        # Save config and tokenizer for later evaluation
        with open(os.path.join(model_dir, "config.pkl"), 'wb') as f:
            pickle.dump(self.config, f)
            
        if not isinstance(self.tokenizer, BertTokenizer):  # Assuming BertTokenizer is imported
            self.tokenizer.save(os.path.join(model_dir, "tokenizer.pkl"))
            
        return model_dir

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, tokenizer=None):
        """Load model from saved checkpoint"""
        model_dir = os.path.dirname(checkpoint_path)
        
        # Load config
        with open(os.path.join(model_dir, "config.pkl"), 'rb') as f:
            config = pickle.load(f)
            
        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
            if os.path.exists(tokenizer_path):
                tokenizer = BaseTokenizer.load(tokenizer_path)  # Assuming BaseTokenizer is imported
        
        # Initialize model
        model = cls(config, tokenizer)
        model.load_state_dict(torch.load(checkpoint_path))
        return model