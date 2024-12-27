if __name__ == "__main__":
    """Inizio funzione clustering."""

    file_path = "../data/question_train.json"  # File di training da clusterizzare
    # File in cui esportare i cluster delle domande sensitive
    sensitive_clusters_file = "export/sensitive_clusters.json"
    # File in cui esportare i cluster delle domande non-sensitive
    non_sensitive_clusters_file = "export/non_sensitive_clusters.json"

    # File in cui esportare le eventuali statistiche dei cluster sensitive
    sensitive_stats_file = "export/sensitive_statistics.json"

    # File in cui esportare le statistiche dei cluster non-sensitive
    non_sensitive_stats_file = "export/non_sensitive_statistics.json"

    # Step 1: Inizio caricamento delle domande
    print("Caricamento delle domande dal JSON...")