from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(features, labels, split=True):
    """
    Addestra un modello Random Forest con parallelizzazione.
    Se split=True, esegue uno split interno su features e labels.
    Se split=False, allena direttamente su features e labels e restituisce None per X_test, y_test.
    """
    if split:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    else:
        X_train, y_train = features, labels
        X_test, y_test = None, None

    # Usa tutti i core disponibili per velocizzare l'allenamento
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, X_test, y_test