import numpy as np

from sklearn.base import clone
from itertools import combinations

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from tqdm import tqdm


## Sequential feature selection algorithms
class SBS():
    ''' Sequential Backward Selection
    You can find a detailed evaluation of several sequential feature algorithms in
    [1] Comparative Study of Techniques for Large-Scale Feature Selection, F. Ferri,
    P. Pudil, M. Hatef, and J. Kittler, pages 403-413, 1994.
    '''
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        #X_train, X_test, y_train, y_test = \
        #    train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        dim = X.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        # score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        score, score_std = self._calc_score_CV(X, y, self.indices_)
        self.scores_ = [score]
        self.scores_std_ = [score_std]

        ps_bar = tqdm(total = dim - self.k_features + 1)
        while dim > self.k_features:
            scores = []
            score_stds = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                # score = self._calc_score(X_train, y_train, X_test, y_test, p)
                score, score_std = self._calc_score_CV(X, y, p)
                scores.append(score)
                score_stds.append(score_std)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)

            self.scores_.append(scores[best])
            self.scores_std_.append(score_stds[best])
            ps_bar.update(1)
            dim -= 1
        ps_bar.close()
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

    def _calc_score_CV(self, X, y, indices):
        #self.estimator.fit(X_train[:, indices], y_train)
        #y_pred = self.estimator.predict(X_test[:, indices])
        #score = self.scoring(y_test, y_pred)
        scores = cross_val_score(estimator=self.estimator, X=X[:, indices], y=y, cv=5)  # , n_jobs=5
        # score = np.mean(scores)
        return np.mean(scores), np.std(scores)
