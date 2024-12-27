import json

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# Funzione per codificare le domande negli embeddings di Sentence-BERT
def encode_questions(questions, model_name='all-MiniLM-L6-v2'):
    """Convert text questions into semantic embeddings using Sentence-BERT."""
    model = SentenceTransformer(model_name)
    return model.encode(questions)

# Funzione per caricare le domande dal file json
def load_questions_from_json(file_path):
    """Load questions and sensitivity labels from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    sensitive = [item["question_en"] for item in data if item["sensitive?"] == 1]
    non_sensitive = [item["question_en"] for item in data if item["sensitive?"] == 0]
    return sensitive, non_sensitive

# Funzione per ridurre la dimensionalità tramite PCA
def reduce_dimensions(embeddings, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


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

    # Step 2 : Elaborazione delle domande sensitive

    sensitive_embeddings = encode_questions(sensitive_questions)
    reduced_sensitive_embeddings = reduce_dimensions(sensitive_embeddings)