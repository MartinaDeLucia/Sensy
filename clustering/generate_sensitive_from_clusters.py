import json
import random


def generate_non_sensitive_json(clusters_file, output_file):
    """
    Genero un JSON file con tutte le domande non sensitive,
    includendo il rumore, e settando "sensitive?": 0.

    """
    # Load the cluster data
    with open(clusters_file, 'r', encoding='utf-8') as file:
        clusters = json.load(file)

    # Prepare the output list
    output_data = []

    # Add all questions from all clusters, including noise
    for cluster_id, questions in clusters.items():
        for question in questions:
            output_data.append({"question_en": question, "sensitive?": 0})

    # Save the output to a JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)

    print(f"Generated JSON saved to {output_file}.")
    return output_data


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

    # Peparo la lista di output
    output_data = []

    # Aggiungo tutte le domande dai cluster senza rumore all'output

    for cluster_id, questions in clusters.items():
        if cluster_id != "-1":  # Exclude noise
            for question in questions:
                output_data.append({"question_en": question, "sensitive?": 1})

    # Seleziono un sottoset delle domande con rumore (la media)
    noise_questions = clusters.get("-1", [])
    selected_noise_questions = random.sample(noise_questions, min(num_noise_questions, len(noise_questions)))

    # Aggiungo le domande con rumore all'output
    for question in selected_noise_questions:
        output_data.append({"question_en": question, "sensitive?": 1})

    # Salvo nel json
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4, ensure_ascii=False)

    print(f"Generated JSON saved to {output_file}.")
    return output_data


if __name__ == "__main__":
    statistics_file = "export/sensitive_statistics.json"  # Path al JSON di statistiche dei cluster sensitive
    clusters_file = "export/sensitive_clusters.json"  # Path al JSON dei cluster sensitive
    output_sensitive_file = "export/question_train_sensitive_clustered_output.json"  # Path all'output

    output_sensitive = generate_sensitive_json(statistics_file, clusters_file, output_sensitive_file)

    clusters_file = "export/non_sensitive_clusters.json"  # Path al JSON dei cluster non-sensitive
    output_nonsensitive_file = "export/question_train_nonsensitive_clustered_output.json"  # Path al JSON di output dei cluster non-sensitive

    output_nonsensitive = generate_non_sensitive_json(clusters_file, output_nonsensitive_file)

    output_total_file = "export/total_clusters_question_train.json"

    total_data = output_nonsensitive + output_sensitive

    with open(output_total_file, 'w', encoding='utf-8') as file:
        json.dump(total_data, file, indent=4, ensure_ascii=False)

    print(f"Generated Total JSON saved to {output_total_file}.")
