import pandas as pd
import nltk
import json

def clean_dataset(file_path):
    """
    Carica e pulisce il dataset da un file JSON.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Converti il JSON in DataFrame
    df = pd.DataFrame(data)
    print(f"Prima della pulizia di {file_path} sono: {df.count()}")

    # Rimuovi duplicati
    #df = df.drop_duplicates()
    #print(f"Dopo la pulizia di {file_path} sono: {df.count()}")

    # Se esiste la colonna 'question', eliminala
    if 'question' in df.columns:
        df = df.drop(columns=['question'])

    # Tokenizzazione del testo in inglese
    df["question_tokenized"] = df["question_en"].apply(nltk.word_tokenize)

    return df