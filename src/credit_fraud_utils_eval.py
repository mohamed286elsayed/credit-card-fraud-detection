
import logging
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)


# Create logs directory
log_dir = Path("src/reports/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=log_dir / "experiments.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_val,
    y_val,
    y_pred,
    y_proba,
    experiment_name,
):
    y_pred = optimized_threshold(model, X_val, y_val)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )
    recall = recall_score(
        y_val, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )
    pr_auc = average_precision_score(
        y_val, y_proba
    )

    logger.info("Experiment: %s", experiment_name)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("F1-score: %.4f", f1)
    logger.info("PR-AUC: %.4f", pr_auc)
    logger.info("-" * 40)

    print(f"Experiment: {experiment_name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")



def optimized_threshold(model, X_val, y_val):
    y_scores = model.predict_proba(X_val)[: ,1]

    precision, recall, threshold = precision_recall_curve(y_val, y_scores)
    precision, recall = precision[:-1], recall[:-1]

    f1 = 2 * (precision * recall) / (precision + recall)
    best_idx = np.argmax(f1)
    best_threshold = threshold[best_idx]

    y_pred = (y_scores >= best_threshold).astype(int)
    return y_pred





