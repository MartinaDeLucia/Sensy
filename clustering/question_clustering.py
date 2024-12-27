import json

import numpy as np
from sklearn.decomposition import PCA
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

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

# Funzione per fare il clustering tramite HDBScan utilizzando la distanza in coseno
def perform_clustering(embeddings, min_cluster_size=50, min_samples=5):
    distance_matrix = cosine_distances(embeddings)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_method='eom'
    )
    return clusterer.fit_predict(distance_matrix)

# Funzione per organizzare le domande in base alle labels dei cluster
def organize_clusters(labels, questions):
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(questions[idx])
    return clusters


# Funzione per generare delle statistiche sui cluster
# Domande totali
# Numero di cluster
# Punti di rumore
# Numero di domande per cluster

def generate_statistics(clusters):
    num_clusters = len(clusters) - (1 if -1 in clusters else 0)  # Exclude noise (-1)
    stats = {
        "total_questions": sum(len(questions) for questions in clusters.values()),
        "number_of_clusters": num_clusters,
        "noise_points": len(clusters[-1]) if -1 in clusters else 0,
        "questions_per_cluster": {str(key): len(value) for key, value in clusters.items()}
    }
    return stats

# Funzione per salvare i cluster in un file JSON
def save_clusters_to_file(clusters, output_file):
    clusters_str_keys = {str(key): value for key, value in clusters.items()}
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(clusters_str_keys, file, indent=4, ensure_ascii=False)
    print(f"Clusters salvati al percorso:  {output_file}")

# Funzione per salvare le statistiche in un file JSON
def save_statistics_to_file(stats, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(stats, file, indent=4, ensure_ascii=False)
    print(f"Statistiche dei cluster salvate al percorso: {output_file}")

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
    print("Elaborando domande sensitive...")
    sensitive_embeddings = encode_questions(sensitive_questions)
    reduced_sensitive_embeddings = reduce_dimensions(sensitive_embeddings)
    sensitive_labels = perform_clustering(reduced_sensitive_embeddings)

    # Step 3 : Elaborazione delle domande non sensitive
    print("Elaborando domande non sensitive...")
    non_sensitive_embeddings = encode_questions(non_sensitive_questions)
    reduced_non_sensitive_embeddings = reduce_dimensions(non_sensitive_embeddings)
    non_sensitive_labels = perform_clustering(reduced_non_sensitive_embeddings)

    #Step 4: Organizzo domande tramite le label dei cluster...
    print("Organizzazione domande...")
    non_sensitive_clusters = organize_clusters(non_sensitive_labels, non_sensitive_questions)
    sensitive_clusters = organize_clusters(sensitive_labels, sensitive_questions)
    non_sensitive_stats = generate_statistics(non_sensitive_clusters)
    sensitive_stats = generate_statistics(sensitive_clusters)
    save_clusters_to_file(non_sensitive_clusters, non_sensitive_clusters_file)
    save_statistics_to_file(non_sensitive_stats, non_sensitive_stats_file)
    print("\nClustering completato per le domande sensitive e non sensitive con successo.")
