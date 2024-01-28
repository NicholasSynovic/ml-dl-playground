from typing import Any, List, Tuple

from numpy import ndarray
from progress.bar import Bar
from sklearn.svm import SVC
from sklearn.svm._base import BaseSVC
from sklearnex.svm import SVC as intelSVC

from ml_dl_playground.utils import hyperparameterTuning

DEFAULT_SVC_PARAMETERS: List[
    dict[str, str | int | float]
] = hyperparameterTuning.gridSearch(
    options={
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "max_iter": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        "random_state": [42],
    }
)


def _trainSVC(
    estimator: BaseSVC,
    trainingData: Tuple[ndarray, ndarray],
    **kwargs,
) -> BaseSVC:
    model: BaseSVC = estimator.set_params(**kwargs)
    model.fit(X=trainingData[0], y=trainingData[1])
    return model


def trainSVC(
    trainingData: Tuple[ndarray, ndarray],
    **kwargs,
) -> SVC:
    return _trainSVC(
        estimator=SVC(),
        trainingData=trainingData,
        **kwargs,
    )


def trainIntelSVC(
    trainingData: Tuple[ndarray, ndarray],
    **kwargs,
) -> intelSVC:
    return _trainSVC(
        estimator=intelSVC(),
        trainingData=trainingData,
        **kwargs,
    )
