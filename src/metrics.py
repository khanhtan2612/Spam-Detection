"""Score function: đánh giá dự đoán bằng confusion matrix (1 = spam, 0 = ham)."""

import numpy as np


def confusion_matrix(y_true, y_pred):
    """Trả [[TN, FP], [FN, TP]]: hàng là nhãn thật (ham, spam), cột là nhãn dự đoán."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(tp, fp, fn):
    """precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2·P·R/(P+R). Mẫu số 0 thì trả 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def classification_scores(y_true, y_pred):
    """Trả accuracy, precision/recall/F1 của lớp spam và macro-F1 = (F1_spam + F1_ham)/2.

    F1 của lớp ham tính như lớp spam nhưng đổi vai: TP_ham = TN, FP_ham = FN, FN_ham = FP.
    """
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision, recall, f1_spam = precision_recall_f1(tp, fp, fn)
    _, _, f1_ham = precision_recall_f1(tn, fn, fp)
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "spam_precision": precision, "spam_recall": recall, "spam_f1": f1_spam,
        "ham_f1": f1_ham, "macro_f1": (f1_spam + f1_ham) / 2,
    }
