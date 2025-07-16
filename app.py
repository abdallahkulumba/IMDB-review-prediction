import os
import yaml
import pickle
import torch
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.preprocessor import TextPreprocessor
from utils.tokenizer import PyTorchTokenizer, BertTokenizer
from configs.model_configs import (
    RNNConfig, LSTMConfig, GRUConfig,
    GloVeConfig, BERTConfig, TransformerConfig
)
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.glove_model import GloVeModel
from models.bert_model import BERTModel
from models.transformer_model import TransformerModel

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 15px;
    }
    .stButton button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #357abd;
        transform: scale(1.02);
    }
    .model-card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        color: #2c3e50;
    }
    .positive {
        color: #27ae60;
        font-weight: bold;
    }
    .negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    models, tokenizers = {}, {}
    preprocessor = TextPreprocessor()

    registry = {
        RNNModel: (RNNConfig, PyTorchTokenizer),
        LSTMModel: (LSTMConfig, PyTorchTokenizer),
        GRUModel: (GRUConfig, PyTorchTokenizer),
        GloVeModel: (GloVeConfig, PyTorchTokenizer),
        BERTModel: (BERTConfig, BertTokenizer),
        TransformerModel: (TransformerConfig, PyTorchTokenizer),
    }

    for ModelCls, (ConfigCls, TokCls) in registry.items():
        name = ModelCls.__name__

        # 1️⃣ load hyperparams
        hp_path = os.path.join(
            ConfigCls().log_dir,
            "model_comparison", name, "hparams.yaml"
        )
        with open(hp_path) as f:
            cfg = ConfigCls(**yaml.safe_load(f))

        # 2️⃣ load checkpoint
        ckpt_path = os.path.join(
            cfg.model_save_dir, "checkpoints", f"{name}-best.ckpt"
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 3️⃣ load tokenizer
        if TokCls is PyTorchTokenizer:
            tok_path = os.path.join(
                cfg.model_save_dir, "tokenizers", f"{name}_tokenizer.pkl"
            )
            if not os.path.exists(tok_path):
                raise FileNotFoundError(f"Missing tokenizer: {tok_path}")
            with open(tok_path, "rb") as f:
                tok = pickle.load(f)
            cfg.vocab_size = tok.vocab_size
        else:
            tok = TokCls(cfg)

        # 4️⃣ instantiate & load the model
        model = ModelCls(cfg, tok)
        model.load_state_dict(state_dict)
        model.eval()

        models[name] = (model, preprocessor)
        tokenizers[name] = tok

    return models, tokenizers

# Load models
models, tokenizers = load_all_models()

def infer_text(review: str):
    df_rows = []
    for name, (model, preproc) in models.items():
        clean = preproc.clean_batch([review])[0]
        tok = tokenizers[name]
        enc = tok.encode([clean])

        batch = (
            {k: torch.tensor(v) for k, v in enc.items()}
            if isinstance(enc, dict)
            else torch.tensor(enc)
        )

        with torch.no_grad():
            logits = model(batch)
            prob = torch.sigmoid(logits).squeeze().item()
            label = "Positive" if prob >= 0.5 else "Negative"

        df_rows.append({
            "Model": name,
            "Sentiment": label,
            "Confidence": prob,
            "Confidence_Display": f"{prob:.2%}"
        })

    return pd.DataFrame(df_rows)

def plot_gauge(score: float, title: str = "Confidence"):
    color = "#27ae60" if score >= 0.5 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score*100,
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0,100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#f8d7da"},
                {"range": [50, 100], "color": "#d4edda"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        },
        number={"suffix": "%", "font": {"size": 24}}
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=40, b=10, l=10, r=10),
        font={"family": "Arial"}
    )
    return fig

def plot_top_words(review: str):
    words = review.lower().split()
    freq = {w: words.count(w) for w in set(words)}
    df = pd.DataFrame(freq.items(), columns=["word", "count"]) \
           .nlargest(10, "count").sort_values("count", ascending=True)
    
    color_scale = px.colors.sequential.Blues_r
    fig = px.bar(
        df, 
        x="count", 
        y="word", 
        orientation='h',
        color="count",
        color_continuous_scale=color_scale,
        title="<b>Top 10 Most Frequent Words</b>",
        labels={"count": "Frequency", "word": "Word"}
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
        height=400,
        showlegend=False,
        title_x=0.5,
        title_font_size=18
    )
    fig.update_traces(marker_line_width=0)
    return fig

def display_model_comparison(df):
    st.subheader("Model Comparison Results")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Enhanced table display
        st.markdown("### Sentiment Analysis Summary")
        styled_df = df.copy()
        styled_df["Sentiment"] = styled_df.apply(
            lambda x: f'<span class="positive">{x["Sentiment"]}</span>' 
            if x["Sentiment"] == "Positive" 
            else f'<span class="negative">{x["Sentiment"]}</span>', 
            axis=1
        )
        styled_df["Confidence"] = styled_df["Confidence_Display"]
        display_df = styled_df[["Model", "Sentiment", "Confidence"]]
        
        st.markdown(
            display_df.to_html(escape=False, index=False), 
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div style="margin-top: 20px; font-size: 14px; color: #7f8c8d;">
            <i>Note: Confidence scores represent the model's certainty in its prediction.</i>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Overall sentiment visualization
        st.markdown("### Overall Sentiment Consensus")
        pos_count = len(df[df["Sentiment"] == "Positive"])
        total = len(df)
        
        fig = go.Figure(go.Pie(
            labels=["Positive", "Negative"],
            values=[pos_count, total - pos_count],
            hole=0.5,
            marker_colors=["#27ae60", "#e74c3c"],
            textinfo="percent+label"
        ))
        fig.update_layout(
            height=250,
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

def sidebar():
    #st.sidebar.image("https://avatars.githubusercontent.com/u/198412563?v=4", width=50)
    st.sidebar.markdown("<h2>Author</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2>Sserujja Abdallah Kulumba</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <p style='color: #2c3e50; font-size: 16px;'>
        AI Researcher | Biomedical & NLP <br>
        Undergraduate EEE Student at <br>
        Islamic University of Technology
    </p>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Quick Links")
    st.sidebar.markdown("""
    - [GitHub Repository](https://github.com/abdallahkulumba/IMDB-review-prediction-using-RNN.git)
    - [ResearchGate Profile](https://www.researchgate.net/profile/Abdallah-Kulumba-Sserujja)
    """)
    
    st.sidebar.markdown("### 📬 Contact")
    st.sidebar.markdown("""
    - ✉️ abdallahkulumba@iut-dhaka.edu
    - 🌐 [Personal Website](#)
    """)
    
    st.sidebar.markdown("### 🔗 Connect")
    st.sidebar.markdown("""
    <div style="display: flex; gap: 10px;">
        <a href="https://www.linkedin.com/in/abdallah-kulumba-sserujja/">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
        <a href="https://github.com/Abdallahkulumba">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
        </a>
        <a href="https://www.facebook.com/share/1CHA7JWTi7/">
            <img src="https://img.shields.io/badge/Facebook-1DA1F2?style=for-the-badge&logo=facebook&logoColor=white" alt="Facebook">
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 12px; color: #7f8c8d; text-align: center;">
        © 2023 Sserujja Abdallah Kulumba<br>
        All rights reserved
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="IMDB Sentiment Analysis Dashboard",
        layout="wide",
        page_icon="🎬"
    )
    
    # Display sidebar
    sidebar()
    
    # Header section
    st.markdown(
        """
        <div style="background-color: #4a90e2; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; text-align: center;">🎬 IMDB Movie Review Sentiment Analysis</h1>
            <p style="color: white; text-align: center; font-size: 16px;">
                Analyze sentiment across multiple NLP models with confidence metrics
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Main content
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Enter Movie Details")
            movie_name = st.text_input(
                "Enter movie name..",
                placeholder="Enter movie name...",
                label_visibility="visible"
            )
            review = st.text_area(
                "Enter your movie review here...",
                height=100,
                placeholder="Paste your movie review here...",
                label_visibility="visible"
            )
            
            analyze_btn = st.button(
                "Analyze Sentiment",
                key="analyze",
                help="Click to analyze the sentiment of your review"
            )
        
        with col2:
            st.markdown("### About This Tool")
            st.markdown("""
            <div class="model-card">
                <p>This dashboard analyzes movie review sentiment using six different NLP models:</p>
                <ul>
                    <li>RNN</li>
                    <li>LSTM</li>
                    <li>GRU</li>
                    <li>GloVe</li>
                    <li>BERT</li>
                    <li>Transformer</li>
                </ul>
                        NOTE: 
                <p>The models were trained on IMDB dataset and for cases of study purposes we trained 
                        for smaller epochs, you can clone the full model setup on github.</p>
            </div>
            """, unsafe_allow_html=True)
    
    if analyze_btn:
        if not review.strip():
            st.warning("Please enter a movie review to analyze.")
        else:
            with st.spinner("Analyzing sentiment across all models..."):
                df = infer_text(review)
                
                # Display results
                display_model_comparison(df)
                
                # Model-specific tabs
                st.markdown("### Detailed Model Analysis")
                tabs = st.tabs(df["Model"].tolist())
                
                for tab, row in zip(tabs, df.to_dict("records")):
                    with tab:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown(f"#### {row['Model']} Prediction")
                            sentiment_class = "positive" if row["Sentiment"] == "Positive" else "negative"
                            st.markdown(
                                f"<h2 style='text-align: center;' class='{sentiment_class}'>"
                                f"{row['Sentiment']}</h2>", 
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"<p style='text-align: center; font-size: 18px;'>"
                                f"Confidence: {row['Confidence_Display']}</p>", 
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            st.plotly_chart(
                                plot_gauge(row["Confidence"]),
                                use_container_width=True
                            )
                
                # Word frequency analysis
                st.markdown("---")
                st.markdown("### Text Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Word Frequency")
                    st.plotly_chart(
                        plot_top_words(review), 
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("#### Review Length Metrics")
                    word_count = len(review.split())
                    char_count = len(review)
                    avg_word_len = char_count / word_count if word_count > 0 else 0
                    
                    st.metric("Word Count", word_count)
                    st.metric("Character Count", char_count)
                    st.metric("Average Word Length", f"{avg_word_len:.1f} characters")

if __name__ == "__main__":
    main()