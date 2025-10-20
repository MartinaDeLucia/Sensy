# models/cross_validate.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score)

def cross_validate_10fold(model_ctor, X, y, random_state=42):
    """
    10-fold Stratified CV su (X, y).
    Ritorna mean/std di: accuracy, precision_macro, recall_macro, f1_macro, roc_auc, pr_auc.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    accs, precs, recs, f1s, rocs, pras = [], [], [], [], [], []

    for tr, te in skf.split(X, y):
        clf = model_ctor()
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te])

        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X[te])[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X[te])
        else:
            y_score = None

        accs.append(accuracy_score(y[te], y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y[te], y_pred, average="macro", zero_division=0)
        precs.append(p); recs.append(r); f1s.append(f1)

        if y_score is not None:
            try: rocs.append(roc_auc_score(y[te], y_score))
            except: pass
            pras.append(average_precision_score(y[te], y_score))

    agg = lambda a: (float(np.mean(a)), float(np.std(a))) if a else (None, None)
    return {
        "accuracy": agg(accs),
        "precision_macro": agg(precs),
        "recall_macro": agg(recs),
        "f1_macro": agg(f1s),
        "roc_auc": agg(rocs),
        "pr_auc": agg(pras),
    }
