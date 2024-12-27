from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, print_report=True):
    """
    Valuta il modello su dati di test.
    Se print_report=False, restituisce solo l'accuratezza.
    Altrimenti stampa il classification report e l'accuratezza.
    """
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    if print_report:
        print("Report di classificazione:")
        print(report)
    return report