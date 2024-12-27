from preprocessing.clean_data import clean_dataset

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

    
    