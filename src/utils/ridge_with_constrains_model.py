import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array


class RidgeWithPositiveConstraints(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, positive_features=None, fit_intercept=True):
        """
        Inicializa o modelo de regressão Ridge com restrições de coeficientes positivos.

        Args:
            alpha (float): Parâmetro de regularização da Ridge Regression.
            positive_features (list): Índices das colunas de X que devem ter coeficientes positivos.
        """
        self.alpha = alpha
        self.positive_features = positive_features
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Ajusta o modelo de regressão Ridge aos dados de treinamento.

        Args:
            X (np.ndarray): Matriz de features.
            y (np.ndarray): Vetor de alvo.

        Returns:
            self: Objeto ajustado.
        """
        X = check_array(X, ensure_2d=True)
        y = check_array(y, ensure_2d=False)
        n_samples, n_features = X.shape

        # Adicionar o intercepto
        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])

        # Variáveis de otimização
        self.beta_ = cp.Variable(n_features + int(self.fit_intercept))

        # Definir a função de perda (OLS + regularização Ridge)
        objective = cp.Minimize(
            cp.sum_squares(X @ self.beta_ - y) + self.alpha * cp.sum_squares(self.beta_)
        )

        # Adicionar restrições de coeficientes positivos para as features especificadas
        constraints = []
        if self.positive_features is not None:
            adjusted_positive_features = [
                pf + 1 if self.fit_intercept else pf for pf in self.positive_features
            ]
            constraints = [self.beta_[i] >= 0 for i in adjusted_positive_features]

        # Resolver o problema de otimização
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if self.beta_.value is None:
            raise ValueError(
                "A solução do problema de otimização falhou. Verifique os dados e as restrições."
            )

        # Armazenar os coeficientes
        self.coef_ = (
            self.beta_.value[1:].reshape(-1, 1)
            if self.fit_intercept
            else self.beta_.value.reshape(-1, 1)
        )
        self.intercept_ = self.beta_.value[0] if self.fit_intercept else 0.0

        return self

    def predict(self, X):
        """
        Faz previsões nos dados fornecidos.

        Args:
            X (np.ndarray): Matriz de features.

        Returns:
            np.ndarray: Previsões.
        """
        X = check_array(X, ensure_2d=True)
        pred = X @ self.coef_ + self.intercept_
        return pred.flatten()

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "positive_features": self.positive_features}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


if __name__ == "__main__":
    # Exemplo de uso
    X = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]])
    y = np.array([1, 2, 3, 4])

    # Suponha que queremos que apenas a feature na coluna 0 tenha coeficiente positivo
    positive_features = [0]

    modelo = RidgeWithPositiveConstraints(
        alpha=1.0, positive_features=positive_features
    )
    modelo.fit(X, y)
    previsoes = modelo.predict(X)

    print("Coeficientes:", modelo.coef_)
    print("Previsões:", previsoes)
