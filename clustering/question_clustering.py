import json

# Funzione per caricare le domande dal file json
def load_questions_from_json(file_path):
    """Load questions and sensitivity labels from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    sensitive = [item["question_en"] for item in data if item["sensitive?"] == 1]
    non_sensitive = [item["question_en"] for item in data if item["sensitive?"] == 0]
    return sensitive, non_sensitive


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

    sensitive_questions, non_sensitive_questions = load_questions_from_json(file_path)
