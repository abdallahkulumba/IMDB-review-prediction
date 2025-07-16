# models/transformer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class TransformerModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config    = config
        self.tokenizer = tokenizer

        # override config from tokenizer
        self.config.vocab_size      = tokenizer.vocab_size
        self.config.max_seq_length  = tokenizer.max_len

        # embeddings & positional encoding
        self.embedding    = nn.Embedding(
                                self.config.vocab_size,
                                self.config.d_model
                            )
        self.pos_encoding = self._positional_encoding(
                                self.config.max_seq_length,
                                self.config.d_model
                            )

        # transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dff,
            dropout=self.config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers
        )

        self.dropout    = nn.Dropout(self.config.dropout_rate)
        self.classifier = nn.Linear(self.config.d_model, 1)

    def _positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        # x is [B, T] where T <= max_seq_length
        emb = self.embedding(x)
        emb = emb + self.pos_encoding[:, :emb.size(1), :].to(emb.device)
        out = self.transformer(emb)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.classifier(out).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss   = F.binary_cross_entropy_with_logits(logits, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss   = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
