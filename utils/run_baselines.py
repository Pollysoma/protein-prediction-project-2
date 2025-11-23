import torch
from torch import optim, nn, utils, Tensor
from safetensors.torch import load_file

import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc,balanced_accuracy_score, precision_score, recall_score

# run baseline models like knn, logistic regression, random forest on the embeddings
def run_models(models, emb_pretrain, emb_posttrain):
    """
    Run a list of specified models on the data and returns results across 5 folds of cross-validation
    Args:
        models (dict): A dictionary where keys are model names and values are instantiated sklearn model objects
        emb_pretrain (dict): A dictionary containing 'embeddings' key with pre-training embeddings as numpy array
        emb_posttrain (dict): A dictionary containing 'embeddings' key with post-training embeddings as numpy array
    Returns:
        results (dict): A dictionary where keys are model names and values are dictionaries with 'trues' and 'preds' lists
    """

    results = {}
    for model_name in models.keys():
        results[model_name] = {'trues':[], 'preds':[]}
        
    X_pre = emb_pretrain # ['embeddings']
    y_pre = np.zeros(X_pre.shape[0])

    X_post = emb_posttrain # ['embeddings']
    y_post = np.ones(X_post.shape[0])
    X = np.concatenate([X_pre, X_post], axis=0)
    y = np.concatenate([y_pre, y_post], axis=0)

    print(f"Total samples: {X.shape[0]}, Pre-train: {X_pre.shape[0]}, Post-train: {X_post.shape[0]}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_num = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_scores = model.predict_proba(X_test)[:, 1]
            results[model_name]['trues'].extend(y_test)
            results[model_name]['preds'].extend(y_scores)
        print(f"Completed fold {fold_num}/5")
        fold_num += 1

    print("All models evaluated.")
    return results


def evaluate_results(results):
    """
    Evaluate the results of models.
    Args:
        results (dict): A dictionary where keys are model names and values are dictionaries with 'trues' and 'preds' lists
    Returns:
        eval_results (dict): A dictionary where keys are model names and values are dictionaries with 'auc', 'fpr', and 'tpr'
    """

    eval_results = {}
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(res['trues'], res['preds'])
        roc_auc = auc(fpr, tpr)

        eval_results[model_name] = {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr}
        
        preds_binary = (np.asarray(res['preds']) >= 0.5).astype(int)
        
        eval_results[model_name]['balanced_accuracy'] = float(balanced_accuracy_score(res['trues'], preds_binary))
        eval_results[model_name]['precision'] = float(precision_score(res['trues'], preds_binary))
        eval_results[model_name]['recall'] = float(recall_score(res['trues'], preds_binary))

    return eval_results
