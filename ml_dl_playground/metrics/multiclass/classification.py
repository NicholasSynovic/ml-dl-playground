from numpy import ndarray
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC


class MulticlassClassificationMetrics:
    def __init__(self, yTrue: ndarray, yPredicted: ndarray) -> None:
        self.yTrue: ndarray = yTrue
        self.yPredicted: ndarray = yPredicted

    def balancedAccuracyScore(self) -> float:
        return balanced_accuracy_score(
            y_true=self.yTrue,
            y_pred=self.yPredicted,
        )
