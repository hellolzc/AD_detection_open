# Stratified GroupKFold - Issue #13621 - scikit-learn/scikit-learn
# https://github.com/scikit-learn/scikit-learn/issues/13621
import numpy as np
from collections import Counter, defaultdict
from sklearn.utils import check_random_state

class RepeatedStratifiedGroupKFold():
    """ Stratified GroupKFold
    """
    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        """ n_splits=5, n_repeats=1, random_state=None
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    def __str__(self):
        return "RepeatedStratifiedGroupKFold(n_splits=%d, n_repeats=%d)" % \
            (self.n_splits, self.n_repeats)
    # Implementation based on this kaggle kernel:
    #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        k = self.n_splits
        def eval_y_counts_per_fold(y_counts, fold):  # 
            """Assess the consistency of each fold distribution and the overall distribution
            Parameters:
                y_counts: array [label1_count, label2_count]
                fold: int
            Uesed value defined outside:
                y_counts_per_fold, k, y_distr
            Return:
                float: Average inconsistency across labels
            """
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts  # Eliminate previous operational impact
            return np.mean(std_per_label)
            
        rnd = check_random_state(self.random_state)
        for _ in range(self.n_repeats):
            labels_num = np.max(y) + 1
            # defaultdict: When the key does not exist but is lookuped, instead of keyError, a default value is returned.
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)
        
            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            # Assign unevenly distributed groups first
            sorted_groups_and_y_counts = sorted(groups_and_y_counts, key=lambda x: -np.std(x[1]))
            for g, y_counts in sorted_groups_and_y_counts:
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(groups) if g in test_groups]

                yield train_indices, test_indices


import matplotlib.pyplot as plt
from sklearn import model_selection

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Draw the train & test data distribution of each fold"""
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                c=indices, marker='_', lw=lw, cmap=plt.cm.coolwarm,
                vmin=-.2, vmax=1.2)

    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_',
            lw=lw, cmap=plt.cm.Paired)
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker='_',
            lw=lw, cmap=plt.cm.tab20c)

    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
        xlabel='Sample index', ylabel="CV iteration",
        ylim=[n_splits+2.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)

def test_example():
    # demonstration
    np.random.seed(1338)
    n_splits = 4
    n_repeats=5

    # Generate the class/group data
    n_points = 100
    X = np.random.randn(100, 10)

    percentiles_classes = [.4, .6]
    y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])

    # Evenly spaced groups
    g = np.hstack([[ii] * 6 for ii in range(18)])[:100]

    fig, ax = plt.subplots(1,2, figsize=(14,4))

    cv_nogrp = model_selection.RepeatedStratifiedKFold(n_splits=n_splits,
                                                    n_repeats=n_repeats,
                                                    random_state=1338)
    cv_grp = RepeatedStratifiedGroupKFold(n_splits=n_splits,
                                        n_repeats=n_repeats,
                                        random_state=1338)

    plot_cv_indices(cv_nogrp, X, y, g, ax[0], n_splits * n_repeats)
    plot_cv_indices(cv_grp, X, y, g, ax[1], n_splits * n_repeats)

    plt.show()


def test_example2():
    import pandas as pd
    df = pd.DataFrame(data={'y':[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
                'x':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                'group':[1,1,2,2,3,3,4,4,5,5,6,7,8,9,10]})
    # print(df)
    sgkf = RepeatedStratifiedGroupKFold(3, 2, random_state=2019)

    for ii, (tr, tt) in enumerate(sgkf.split(X=df.values, y=df.y.values, groups=df.group.values)):
        print(tr, tt)


if __name__ == "__main__":
    test_example2()
