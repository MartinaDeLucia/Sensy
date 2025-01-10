from models.evaluate_model import evaluate_model
from models.train_model import train_model
from preprocessing.clean_data import clean_dataset
import pandas as pd
import pickle
import json

from preprocessing.feature_extraction import extract_features

if __name__ == "__main__":
    # Configurazione hardcoded
    # questo usa un test-set in un file a parte.
    TRAIN_RATIO = 1.0  # Percentuale di dati di training principale da usare
    TEST_RATIO = 1.0   # Percentuale di dati di test da usare
    ADDITIONAL_TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]  # Percentuali del dataset aggiuntivo

    train_path = "clustering/export/total_clusters_question_train.json"
    additional_train_path = "data/normalized_chatgpt_questions.json"
    test_path = "data/question_test.json"

    print("Caricamento e pulizia del dataset di training principale...")
    train_data = clean_dataset(train_path)

    print("Caricamento e pulizia del dataset di test...")
    test_data = clean_dataset(test_path)

    print("Caricamento e pulizia del dataset di training aggiuntivo...")
    additional_train_data = clean_dataset(additional_train_path)

    # Risultati finali
    results = []

    for ratio in ADDITIONAL_TRAIN_RATIOS:
        REPORT_PATH = f"samples/report/report_squaretestset_{ratio}_new.json"
        print(f"\n==> Inizio allenamento con ADDITIONAL_TRAIN_RATIO={ratio}...\n")

        # Suddivisione in base alla classe per il dataset aggiuntivo
        df_add_zero = additional_train_data[additional_train_data["sensitive?"] == 0]
        df_add_one = additional_train_data[additional_train_data["sensitive?"] == 1]

        min_add_count = min(len(df_add_zero), len(df_add_one))
        df_add_zero_bal = df_add_zero.sample(min_add_count, random_state=42)
        df_add_one_bal = df_add_one.sample(min_add_count, random_state=42)
        df_add_balanced = pd.concat([df_add_zero_bal, df_add_one_bal], ignore_index=True)
        df_add_balanced = df_add_balanced.sample(frac=ratio, random_state=42).reset_index(drop=True)

        # Combinare il dataset principale bilanciato con il dataset aggiuntivo
        df_train_final = pd.concat([train_data, df_add_balanced], ignore_index=True)

        # Estrazione delle feature
        print("Estrazione delle feature dal training set combinato...")
        train_features, train_labels = extract_features(df_train_final)

        print("Estrazione delle feature dal test set...")
        test_features, test_labels = extract_features(test_data)

        # Addestramento del modello
        print("Addestramento del modello sul training set combinato...")
        model, _, _ = train_model(train_features, train_labels, split=False)

        # Valutazione sul test set
        print("Valutazione del modello sul test set bilanciato...")
        report = evaluate_model(model, test_features, test_labels, print_report=True)

        # Salvataggio del risultato
        accuracy = report["accuracy"]
        print(f"Accuratezza con ADDITIONAL_TRAIN_RATIO={ratio}: {accuracy:.4f}")

        # Salvataggio del modello
        model_path = f"samples/models/model_square_testset_ratio_{ratio}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Salvataggio del risultato
        results.append({"additional_train_ratio": ratio, "accuracy": accuracy})

        print(f"Salvataggio del report in {REPORT_PATH}...")
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)

        # Salvataggio dei risultati
    results_df = pd.DataFrame(results)
    results_csv_path = "samples/results/training_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    print("\n==> Tutti i modelli addestrati e risultati salvati con successo!")
              


              

    
    