from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import tqdm
import numpy as np
import pandas as pd


class ModelTrainer:
    def __init__(self, model_list) -> None:
        self.model_list = model_list

    def fit_and_evaluate(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        metrics: str,
        n_splits=10,
        n_repeats=3,
    ) -> pd.DataFrame:
        baseline_results = pd.DataFrame(
            columns=["model_name", f"{metrics}_train_cv", f"{metrics}_val"]
        )
        for idx in tqdm.tqdm(range(len(self.model_list))):

            clf = self.model_list[idx]

            metrics_train = np.round(
                np.mean(
                    cross_val_score(
                        clf,
                        X_train,
                        y_train,
                        scoring=metrics,
                        cv=RepeatedStratifiedKFold(
                            n_splits=n_splits, n_repeats=n_repeats, random_state=2024
                        ),
                        n_jobs=5,
                    )
                ),
                3,
            )
            # test on val_set
            clf.fit(X_train, y_train)
            y_pred_val = clf.predict_proba(X_val)[:, 1]
            metrics_val = self.cal_metrics(y_val, y_pred_val)
            baseline_results.loc[len(baseline_results)] = [
                clf.__class__.__name__,
                metrics_train,
                metrics_val,
            ]
        return baseline_results.sort_values(
            by=f"{metrics}_val", ascending=False
        ).set_index("model_name")

    def cal_metrics(self, y, y_pred) -> float:
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        return auc(fpr, tpr)
