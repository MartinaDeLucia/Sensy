import nltk
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Caricamento sentiment analyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Device (CPU/MPS/GPU) da usare in tutta l'app
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Inizializza tokenizer e modello UNA volta sola
_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_bert_model = BertModel.from_pretrained("bert-base-uncased")
_bert_model.to(device)
_bert_model.eval()

def shared_count_pos_tags(text, pos_tags):
    """
    Conta quanti token in 'text' appartengono alle pos_tags specificate.
    """
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return sum(1 for _, tag in tagged if tag in pos_tags)

def shared_count_sensitive_words(text, sensitive_words):
    """
    Conta quante parole di 'text' (in minuscolo) appartengono alla lista 'sensitive_words'.
    """
    tokens = nltk.word_tokenize(text.lower())
    return sum(1 for token in tokens if token in sensitive_words)

def shared_generate_single_embedding(text):
    """
    Restituisce l'embedding [CLS] (shape: (1, 768)) per una singola frase.
    """
    inputs = _tokenizer(str(text), return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

def shared_generate_batch_embeddings(texts, batch_size=32):
    """
    Restituisce un array numpy (shape: (N, 768)) contenente le embedding [CLS]
    per un batch di frasi di dimensione variabile.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Converte in stringhe
        batch_texts = [str(t) for t in batch_texts]

        inputs = _tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)