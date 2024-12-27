from preprocessing.clean_data import clean_dataset
import pandas as pd

if __name__ == "__main__":
    # Configurazione hardcoded
    # questo usa un test-set in un file a parte.
    TRAIN_RATIO = 1.0  # Percentuale di dati di training principale da usare
    TEST_RATIO = 1.0   # Percentuale di dati di test da usare
    ADDITIONAL_TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]  # Percentuali del dataset aggiuntivo

    train_path = "data/question_train_clustered_output.json"
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
              


              

    
    