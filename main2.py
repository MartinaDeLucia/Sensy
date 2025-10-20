from models.evaluate_model import evaluate_model
from models.train_model import train_model
from preprocessing.clean_data import clean_dataset
from preprocessing.feature_extraction import extract_features
from models.cross_validate import cross_validate_10fold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pandas as pd
import pickle
import json
import numpy as np
import os

# ======================
# CONFIG
# ======================
# Modalità: True => 10-fold CV (ignora TEST); False => esperimento con ratio su TEST
RUN_CV10 = True

# Percorsi dataset (niente clustering/)
TRAIN_PATH      = "data/dataset_SensY.json"
ADDITIONAL_PATH = "data/normalized_chatgpt_questions.json"  # es: "data_improvement/dataset_chatbot_arena.json"
TEST_PATH       = "data/question_test.json"                 # usato solo se RUN_CV10=False

# Output
REPORT_DIR  = "samples/report"
MODEL_DIR   = "samples/models"
ERRORS_DIR  = "samples/errors"
RESULTS_DIR = "samples/results"

# Esperimento ratio (usato solo se RUN_CV10=False)
ADDITIONAL_TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def rf_ctor():
    # stesso modello del tuo train
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


if __name__ == "__main__":
    ensure_dir(REPORT_DIR); ensure_dir(MODEL_DIR); ensure_dir(ERRORS_DIR); ensure_dir(RESULTS_DIR)

    print("Caricamento (senza pulizia, senza bilanciamento)...")
    df_train = clean_dataset(TRAIN_PATH)
    df_add   = clean_dataset(ADDITIONAL_PATH)

    if RUN_CV10:
        # ====== MODALITÀ CV 10-FOLD ======
        print("\n=== Modalità: 10-FOLD STRATIFIED CROSS-VALIDATION ===")
        #df_tr_all = pd.concat([df_train, df_add], ignore_index=True)
        print(f"Train+Additional: {len(df_train)} righe")

        print("Estrazione feature (tutte, come prima)...")
        X_tr, y_tr = extract_features(df_train)
        X_tr = np.asarray(X_tr); y_tr = np.asarray(y_tr, dtype=int)

        metrics = cross_validate_10fold(rf_ctor, X_tr, y_tr, random_state=42)

        out_json = os.path.join(REPORT_DIR, "cv10_report.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n>> Report 10-fold salvato in: {out_json}\n")
        for k, (m, s) in metrics.items():
            print(f"{k:16s}: {('n/a' if m is None else f'mean={m:.4f}  std={s:.4f}')}")
    else:
        # ====== MODALITÀ ESPERIMENTO SU RATIO (hold-out su TEST) ======
        print("\n=== Modalità: Esperimento su ADDITIONAL_TRAIN_RATIOS ===")
        df_test = clean_dataset(TEST_PATH)
        results = []

        for ratio in ADDITIONAL_TRAIN_RATIOS:
            print(f"\n==> Inizio allenamento con ADDITIONAL_TRAIN_RATIO={ratio}...\n")

            # sottoinsieme additional (solo sampling casuale, nessun bilanciamento)
            df_add_subset = df_add.sample(frac=ratio, random_state=42).reset_index(drop=True)

            # combinazione train + additional
            df_train_final = pd.concat([df_train, df_add_subset], ignore_index=True)

            print("Estrazione feature (TRAIN combinato)...")
            X_tr, y_tr = extract_features(df_train_final)
            X_tr = np.asarray(X_tr); y_tr = np.asarray(y_tr, dtype=int)

            print("Estrazione feature (TEST)...")
            X_te, y_te = extract_features(df_test)
            X_te = np.asarray(X_te); y_te = np.asarray(y_te, dtype=int)

            print("Addestramento modello...")
            model, _, _ = train_model(X_tr, y_tr, split=False)

            print("Valutazione su TEST...")
            report = evaluate_model(model, X_te, y_te, print_report=True)
            accuracy = report.get("accuracy", float("nan"))
            print(f"Accuratezza con ADDITIONAL_TRAIN_RATIO={ratio}: {accuracy:.4f}")

            # salvataggi
            model_path  = os.path.join(MODEL_DIR,  f"model_ratio_{ratio}.pkl")
            report_path = os.path.join(REPORT_DIR, f"report_ratio_{ratio}.json")

            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)

            # errori di classificazione
            y_pred = model.predict(X_te)
            error_indices = np.where(y_pred != y_te)[0]
            error_data = df_test.iloc[error_indices].copy()
            error_data["true_label"] = y_te[error_indices]
            error_data["predicted_label"] = y_pred[error_indices]
            error_data["ratio"] = ratio

            error_csv_path = os.path.join(ERRORS_DIR, f"errors_ratio_{ratio}.csv")
            error_data[["question_en", "true_label", "predicted_label", "ratio"]].to_csv(error_csv_path, index=False)
            print(f"Errori salvati in: {error_csv_path}")

            results.append({"additional_train_ratio": ratio, "accuracy": accuracy})

        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(RESULTS_DIR, "training_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"\n==> Tutti i modelli addestrati. Risultati in: {results_csv_path}")
