import pickle
import os
from extract_single import extract_features_single

if __name__ == "__main__":
    samples_dir = "samples/models"
    best_model_path = os.path.join(samples_dir, "model_square_testset_ratio_1.0.pkl")

    # Carichiamo solo il modello, poiché non utilizziamo più TF-IDF e BoW
    with open(best_model_path, "rb") as f:
        model = pickle.load(f)

    # Chiediamo una domanda all'utente
    question = input("Inserisci la tua domanda (in inglese): ")

    # Estraiamo le feature dalla singola domanda (solo sintattiche + BERT)
    features = extract_features_single(question)  # Non serve più passare vectorizzatori

    # Predizione
    prediction = model.predict(features)

    # Stampa del risultato
    if prediction[0] == 1:
        print("La domanda è considerata SENSITIVE.")
    else:
        print("La domanda NON è considerata sensitive.")