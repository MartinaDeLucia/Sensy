import json
import random

def generate_sensitive_json(statistics_file, clusters_file, output_file):
    """
    Genero un JSON file contenente tutte le domande dai cluster e un sottoset di domande dal rumore
    (Questo perchè le domande classificate come noise sono troppe, di conseguenza abbiamo optato
    per prenderci tutti i cluster cosi come sono e utilizziamo la media delle domande di tutti i cluster
    per avere una percentuale di domande dal noise da prendere)

    :param statistics_file: Path del JSON file con le statistiche di clustering
    :param clusters_file: Path del JSON file con i cluster e le domande
    :param output_file: Path dove salvare il nuovo JSON con le domande sensitive clusterizzate da usare
    poi successivamente per l'addestramento
    """

    with open(statistics_file, 'r', encoding='utf-8') as file:
        statistics = json.load(file)

    with open(clusters_file, 'r', encoding='utf-8') as file:
        clusters = json.load(file)

    # Calcolo le domande totali nei cluster escludendo il rumore e il numero di cluster
    questions_per_cluster = statistics["questions_per_cluster"]
    total_clusters = len(questions_per_cluster) - 1
    total_clustered_questions = sum(
        count for cluster_id, count in questions_per_cluster.items() if cluster_id != "-1"
    )
    num_noise_questions = total_clustered_questions // total_clusters



if __name__ == "__main__":
    statistics_file = "export/sensitive_statistics.json"  # Path al JSON di statistiche
    clusters_file = "export/sensitive_clusters.json"  # Path al JSON dei cluster
    output_file = "export/question_train_sensitive_clustered_output.json"  # Path all'output

    generate_sensitive_json(statistics_file, clusters_file, output_file)