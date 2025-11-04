from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
# True labels
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
# Predicted labels
y_pred = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
# Predicted probabilities (for log loss and AUC)
y_proba = [0.9, 0.2, 0.8, 0.4, 0.3, 0.1, 0.7, 0.6, 0.95, 0.2]

# 1. (Overall correctness)Fraction of correct predictions.
print(accuracy_score(y_true, y_pred))

# 2. (Quality of positive predictions) Of all predicted positives,
# how many were actually positive? Formula: TP / (TP + FP)
print(precision_score(y_true, y_pred))

# 3. (Coverage of actual positives) Of all actual positives,
# how many did we correctly predict? Formula: TP / (TP + FN)
print(recall_score(y_true, y_pred))

# 4. (Balance of precision & recall)Harmonic mean of precision and recall.
# Example: F1 = 2 * (Precision * Recall) / (Precision + Recall)
print(f1_score(y_true, y_pred))

# 5. (Breakdown of prediction outcomes)
# Breakdown of prediction outcomes: TN, FP, FN, TP
print(confusion_matrix(y_true, y_pred))

# 6. (Confidence penalty)
print(log_loss(y_true, y_proba))

# 7. (Class separation)Measures how well the model separates classes.
# Range: 0.5 (random) to 1.0 (perfect).
print(roc_auc_score(y_true, y_proba))

