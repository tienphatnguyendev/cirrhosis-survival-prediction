from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5, handling_type=None):
        self.factor = factor
        self.handling_type = handling_type
        self.lower_bound = []
        self.upper_bound = []

    def _outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X: np.ndarray, y=None):
        self.lower_bound = []
        self.upper_bound = []

        X.apply(self._outlier_detector)
        self.feature_names_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if self.handling_type == "cap":
            for i,col in enumerate(X.columns):
                df.loc[df[col] < self.lower_bound[i], col] = self.lower_bound[i]

                df.loc[df[col] > self.upper_bound[i], col] = self.upper_bound[i]

        else:
            for i in range(df.shape[1]):
                x = df.iloc[:, i].copy()
                x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
                df.iloc[:, i] = x
            
        self.columns = df.columns
        return df

    def get_feature_names_out(self, feature_names):
        return [col for col in self.columns]
