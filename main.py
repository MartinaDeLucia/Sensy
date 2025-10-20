from models.evaluate_model import evaluate_model
from models.train_model import train_model
from preprocessing.clean_data import clean_dataset
from preprocessing.feature_extraction import extract_features
import pandas as pd
import pickle
import json
import numpy as np
import os

from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Configurazione
    TRAIN_RATIO = 1.0
    TEST_RATIO = 1.0
    ADDITIONAL_TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]

    train_path = "clustering/export/total_clusters_question_train.json"
    additional_train_path = "data/normalized_chatgpt_questions.json"  # o "data_improvement/dataset_chatbot_arena.json" "data/normalized_chatgpt_questions.json"
    test_path =  "data_improvement/dataset_chatbot_arena.json" # o "data_improvement/dataset_chatbot_arena.json" "data/question_test.json"

    os.makedirs("samples/errors", exist_ok=True)

    print("Caricamento e pulizia del dataset di training principale...")
    train_data = clean_dataset(train_path)

    print("Caricamento e pulizia del dataset di test...")
    test_data = clean_dataset(test_path)

    print("Caricamento e pulizia del dataset di training aggiuntivo...")
    additional_train_data = clean_dataset(additional_train_path)

    # Bilancia il dataset aggiuntivo se necessario
    #label_counts = additional_train_data["sensitive?"].value_counts()
    #if 0 in label_counts and 1 in label_counts and abs(label_counts[0] - label_counts[1]) > 100:
    #    print(f"Il dataset aggiuntivo è sbilanciato: {label_counts.to_dict()}")
    #    print("Bilanciamento in corso...")

#        df_add_zero = additional_train_data[additional_train_data["sensitive?"] == 0]
 #       df_add_one = additional_train_data[additional_train_data["sensitive?"] == 1]
#
 #       min_len = min(len(df_add_zero), len(df_add_one))
  #      df_add_zero_bal = df_add_zero.sample(min_len, random_state=42)
   #     df_add_one_bal = df_add_one.sample(min_len, random_state=42)

    #    additional_train_data = pd.concat([df_add_zero_bal, df_add_one_bal], ignore_index=True).sample(frac=1, random_state=42)
    #else:
     #   print("Il dataset aggiuntivo è già bilanciato.")

    results = []

    for ratio in ADDITIONAL_TRAIN_RATIOS:
        REPORT_PATH = f"samples/report/report_squaretestset_{ratio}_new.json"
        print(f"\n==> Inizio allenamento con ADDITIONAL_TRAIN_RATIO={ratio}...\n")

        # Estrazione sottoinsieme del dataset aggiuntivo
        df_add_balanced = additional_train_data.sample(frac=ratio, random_state=42).reset_index(drop=True)

        # Combinazione dei dataset
        df_train_final = pd.concat([train_data, df_add_balanced], ignore_index=True)

        print("Estrazione delle feature dal training set combinato...")
        train_features, train_labels = extract_features(df_train_final)

        print("Estrazione delle feature dal test set...")
        test_features, test_labels = extract_features(test_data)

        print("Addestramento del modello sul training set combinato...")
        model, _, _ = train_model(train_features, train_labels, split=False)

        print("Valutazione del modello sul test set bilanciato...")
        report = evaluate_model(model, test_features, test_labels, print_report=True)

        accuracy = report["accuracy"]
        print(f"Accuratezza con ADDITIONAL_TRAIN_RATIO={ratio}: {accuracy:.4f}")

        # Salvataggio modello
        model_path = f"samples/models/model_square_testset_ratio_{ratio}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Salvataggio report
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)

        # Salvataggio errori
        y_pred = model.predict(test_features)
        error_indices = np.where(y_pred != test_labels)[0]
        error_data = test_data.iloc[error_indices].copy()
        error_data["true_label"] = test_labels[error_indices]
        error_data["predicted_label"] = y_pred[error_indices]
        error_data["ratio"] = ratio

        error_csv_path = f"samples/errors/errors_ratio_{ratio}.csv"
        error_data[["question_en", "true_label", "predicted_label", "ratio"]].to_csv(error_csv_path, index=False)
        print(f"Errori salvati in: {error_csv_path}")

        results.append({"additional_train_ratio": ratio, "accuracy": accuracy})

    results_df = pd.DataFrame(results)
    results_csv_path = "samples/results/training_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    print("\n==> Tutti i modelli addestrati e risultati salvati con successo!")