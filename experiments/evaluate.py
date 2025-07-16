# experiments/evaluate.py

import os
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs.base_config      import BaseConfig
from configs.model_configs    import (
    RNNConfig, LSTMConfig, GRUConfig, GloVeConfig, BERTConfig, TransformerConfig
)
from models.rnn_model         import RNNModel
from models.lstm_model        import LSTMModel
from models.gru_model         import GRUModel
from models.glove_model       import GloVeModel
from models.bert_model        import BERTModel
from utils.tokenizer          import PyTorchTokenizer, BertTokenizer
from utils.data_loader        import TextDataset
from utils.evaluator          import ModelEvaluator
from experiments.train        import load_data
from models.transformer_model import TransformerModel



def evaluate_all_models(config: BaseConfig) -> pd.DataFrame:
    """
    Evaluate every checkpoint in model_save_dir/checkpoints using
    the same data pipeline and ModelEvaluator used in training.
    """
    # 1) Load train/test split to get test_df (and train_df for tokenizer fitting)
    train_df, test_df = load_data(config)

    # 2) Gather .ckpt files
    ckpt_dir = os.path.join(config.model_save_dir, "checkpoints")
    ckpt_paths = glob.glob(os.path.join(ckpt_dir, "*-best.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}. Run with --train first.")

    # 3) Map keys → (ModelClass, ConfigClass) as in train.py
    model_map = {
        #'RNN':   (RNNModel,   RNNConfig),
        #'LSTM':  (LSTMModel,  LSTMConfig),
        #'GRU':   (GRUModel,   GRUConfig),
        #'GloVe': (GloVeModel, GloVeConfig),
        #'BERT':  (BERTModel,  BERTConfig),
        'Transformer': (TransformerModel, TransformerConfig),
    }
    # Build a reverse lookup: LightningModule class-name → (ModelClass, ConfigClass)
    name2map = {
        cls.__name__: (cls, cfg)
        for cls, cfg in model_map.values()
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_metrics = []

    for ckpt_path in ckpt_paths:
        fname = os.path.basename(ckpt_path)
        raw_name = fname.split('-')[0]  # e.g. "BERTModel", "GloVeModel"
        print(f"\n→ Evaluating {raw_name} from {fname}")

        entry = name2map.get(raw_name)
        if entry is None:
            print(f"   Skipping unknown model {raw_name}")
            continue
        ModelCls, ConfigCls = entry

        # 4) Instantiate config & tokenizer
        model_cfg = ConfigCls()
        if raw_name == BERTModel.__name__:
            tokenizer = BertTokenizer(model_cfg)
        else:
            tokenizer = PyTorchTokenizer(model_cfg)
            tokenizer.fit(train_df['text'].tolist())

        # 5) Instantiate the model and load its checkpoint
        model = ModelCls(model_cfg, tokenizer).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        # 6) Build DataLoader for the test split
        test_ds = TextDataset(
            texts=test_df['text'].values,
            labels=test_df['label'].values,
            tokenizer=model.tokenizer
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=model_cfg.batch_size,
            shuffle=False,
            num_workers=0
        )

        # 7) Evaluate with ModelEvaluator
        evaluator = ModelEvaluator(model)
        pred_probs, true_labels = evaluator.generate_predictions(test_loader)
        metrics = evaluator.calculate_metrics(true_labels, pred_probs)
        metrics['model'] = raw_name
        all_metrics.append(metrics)

        # 8) Save per-model metrics & artifacts
        outdir = os.path.join(config.results_dir, raw_name)
        os.makedirs(outdir, exist_ok=True)
        evaluator.save_metrics(metrics, os.path.join(outdir, 'metrics.csv'))
        evaluator.generate_confusion_matrix(
            true_labels,
            pred_probs,
            save_path=os.path.join(outdir, 'confusion_matrix.png')
        )
        report = evaluator.generate_classification_report(true_labels, pred_probs)
        report.to_csv(os.path.join(outdir, 'classification_report.csv'), index=False)

    # 9) Combine all metrics into a summary DataFrame
    results_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(config.results_dir, 'evaluation_results.csv')
    results_df.to_csv(summary_path, index=False)
    print(f"\nOverall evaluation saved to {summary_path}\n")

    return results_df
