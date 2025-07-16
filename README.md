---
sdk: streamlit
sdk_version: 1.25.0
---

# 🎬 IMDB Sentiment Analysis Dashboard

A multi-model sentiment analysis dashboard built with Streamlit and PyTorch. This app lets you paste any movie review and instantly compare predictions across multiple deep learning architectures: RNN, LSTM, GRU, GloVe, and Transformer.  This projects falls in the NLP model series where different NLP more tasted to give movie review basing on user's comment
you can access the full file on my hugging face page (https://huggingface.co/Sserujja)

![Dashboard Preview](image/capture.png)


👉 **Live demo:** [Open the app on Hugging Face Spaces](https://huggingface.co/spaces/Sserujja/imdb-sentiment-model)

---

## ✨ Features

- Interactive text input with confidence gauges for each model
- Clean dashboard UI with visual word-frequency analysis
- Lightweight inference-only deployment (no raw data required)
- Modular design for easy model integration
- Hosted on Hugging Face Spaces — accessible and shareable

---

## 🧠 Models Supported

| Model        | Tokenizer       | Checkpoint Format |
|--------------|------------------|-------------------|
| RNN          | PyTorchTokenizer | `.ckpt`           |
| LSTM         | PyTorchTokenizer | `.ckpt`           |
| GRU          | PyTorchTokenizer | `.ckpt`           |
| GloVe        | PyTorchTokenizer | `.ckpt` (no .txt) |
| Transformer  | PyTorchTokenizer | `.ckpt`           |

> BERT model was trained but not included in deployment to reduce size.

![Dashboard Preview](image/capture2.png)


---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Sserujja/imdb-sentiment-model.git
cd imdb-sentiment-model

# 2. Create environment and install requirements
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
