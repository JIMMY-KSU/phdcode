import numpy as np


class ComplexLinearRegression:
    """
    Fit a linear regression model where the data is allowed to be complex.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, Xin, y):
        X_offset = np.average(Xin, axis=0)
        X = Xin - X_offset
        y_offset = np.average(y, axis=0)
        y = y - y_offset

        self.coef_, self._residues, self.rank_, self.singular_ = np.linalg.lstsq(X, y)
        self.coef_ = self.coef_.conj().T

        if self.fit_intercept:
            self.coef_ = self.coef_
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.conj().T)
        else:
            self.intercept_ = 0.

    def predict(self, Xin):
        return np.dot(Xin, self.coef_.conj().T) + self.intercept_
