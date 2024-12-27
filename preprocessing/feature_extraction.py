from transformers import BertTokenizer, BertModel
import nltk
import torch
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

# Istanzia il SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Lista di parole sensibili
SENSITIVE_WORDS = [
    "war", "violence", "terrorism", "racism", "sexism", "discrimination",
    "abortion", "religion", "politics", "LGBTQ", "poverty", "inequality",
    "slavery", "abuse", "murder", "assault", "genocide", "immigration",
    "gun", "shooting", "protest", "riot", "extremism", "corruption",
    "feminism", "oppression", "hate", "harassment",
    "torture", "massacre", "bombing", "hostage", "kidnapping", "execution",
    "lynching", "cruelty", "bloodshed", "atrocity", "militia", "paramilitary",
    "landmine", "nuclear", "bioweapon", "chemical_weapon", "airstrike", "firing_squad",
    "homophobia", "transphobia", "antisemitism", "islamophobia", "xenophobia",
    "ageism", "ableism", "bigotry", "misogyny", "misandry", "ethnic_cleansing",
    "white_supremacy", "neo_nazi", "kkk", "segregation",
    "apartheid", "junta", "coup", "fascism", "authoritarianism", "dictatorship",
    "martial_law", "censorship", "propaganda", "repression", "surveillance",
    "blacklist", "genocidal",
    "human_trafficking", "forced_marriage", "child_labor", "child_soldier",
    "female_genital_mutilation", "honor_killing", "dowry_death", "bride_burning",
    "acid_attack", "domestic_violence", "pedophilia", "sexual_exploitation",
    "rape", "incest", "molestation", "stalking",
    "self_harm", "suicide", "depression", "anorexia", "bulimia", "overdose",
    "drug_cartel", "drug_lord", "opioid", "heroin", "cocaine", "methamphetamine",
    "mafia", "organized_crime", "cartel", "gang_violence", "money_laundering",
    "racketeering", "human_smuggling",
    "blasphemy", "heresy", "sectarian", "jihad", "fatwa", "religious_persecution",
    "totalitarianism", "apartheid", "ethnostate", "gerrymandering", "coup_d_etat",
    "political_prisoner", "dissident", "black_op", "state_sponsored_terror",
    "hate_speech", "racial_slur", "nazi_symbol", "holocaust_denial", "ethnic_slur",
    "forced_sterilization", "mass_grave", "ethnic_tension", "hate_crime",
    "radicalization", "extremist_cell", "isis", "al_qaeda", "white_nationalism",
    "sex_trafficking", "child_pornography", "sexual_slavery", "coercion", "grooming",
    "sweatshop", "bonded_labor", "indentured_servitude",
    "systemic_racism", "institutional_discrimination", "social_exclusion",
    "marginalization", "caste_system",
    "human_rights_abuse", "war_crime", "crime_against_humanity", "re-education_camp",
    "concentration_camp", "child_abuse", "forced_displacement", "refugee_crisis"
]
def extract_features(data):
    """
    Estrae feature sintattiche (senza num_words) + sentiment + embedding BERT.
    Non utilizza TF-IDF, BoW.
    """
    # Assicuriamoci che question_en sia stringa
    data["question_en"] = data["question_en"].fillna("").astype(str)

    data["num_unique_words"] = data["question_tokenized"].apply(lambda x: len(set(x)))
    data["num_verbs"] = data["question_en"].apply(lambda x: count_pos_tags(x, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]))
    data["num_adjectives"] = data["question_en"].apply(lambda x: count_pos_tags(x, ["JJ", "JJR", "JJS"]))
    data["num_nouns"] = data["question_en"].apply(lambda x: count_pos_tags(x, ["NN", "NNS", "NNP", "NNPS"]))
    data["num_sensitive_words"] = data["question_en"].apply(count_sensitive_words)

    # Calcolo del sentiment (compound)
    data["sentiment"] = data["question_en"].apply(lambda x: sia.polarity_scores(x)["compound"])

    # BERT embeddings
    bert_embeddings = generate_bert_embeddings(data["question_en"])

    # Feature finali: num_unique_words, num_verbs, num_adjectives, num_nouns, num_sensitive_words, sentiment + BERT
    syntactic_semantic_features = data[
        ["num_unique_words", "num_verbs", "num_adjectives", "num_nouns", "num_sensitive_words", "sentiment"]
    ].values

    features = np.hstack([syntactic_semantic_features, bert_embeddings])
    labels = data["sensitive?"].values

    return features, labels


def count_pos_tags(text, pos_tags):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return sum(1 for word, tag in tagged if tag in pos_tags)

def count_sensitive_words(text):
    tokens = nltk.word_tokenize(text.lower())
    return sum(1 for token in tokens if token in SENSITIVE_WORDS)
def generate_bert_embeddings(texts, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Assicurati che siano stringhe
        batch_texts = [str(t) for t in batch_texts]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Estrazione della [CLS] embedding per ogni frase (senza mean)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)

    embeddings = np.vstack(embeddings)
    return embeddings