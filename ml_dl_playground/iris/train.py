from typing import Any, List, Tuple

from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm._base import BaseSVC
from sklearnex.svm import SVC as intelSVC


def _trainSVC(
    model: Any,
    data: Tuple[ndarray, ndarray, ndarray, ndarray],
    epochs: List[int] = [1, 10, 50, 100, 200, 500, 1000],
    randomState: int = 42,
) -> GridSearchCV:
    xTrain: ndarray = data[0]
    yTrain: ndarray = data[2]

    parameterGrid: dict[str, List[str | int | float] | int] = {
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "max_iter": epochs,
        "random_state": [randomState],
    }

    gs: GridSearchCV = GridSearchCV(
        estimator=model,
        param_grid=parameterGrid,
        refit=True,
        cv=10,
        verbose=10,
        return_train_score=True,
    )

    gs.fit(X=xTrain, y=yTrain)

    return gs


def trainSVC(
    data: Tuple[ndarray, ndarray, ndarray, ndarray],
    epochs: List[int] = [1, 10, 50, 100, 200, 500, 1000],
    randomState: int = 42,
) -> GridSearchCV:
    return _trainSVC(
        model=SVC(),
        data=data,
        epochs=epochs,
        randomState=randomState,
    )


def trainIntelSVC(
    data: Tuple[ndarray, ndarray, ndarray, ndarray],
    epochs: List[int] = [1, 10, 50, 100, 200, 500, 1000],
    randomState: int = 42,
) -> GridSearchCV:
    return _trainSVC(
        model=intelSVC(),
        data=data,
        epochs=epochs,
        randomState=randomState,
    )


def score(
    gs: GridSearchCV,
    data: Tuple[ndarray, ndarray, ndarray, ndarray],
) -> DataFrame:
    xTest: ndarray = data[1]
    yTest: ndarray = data[3]

    model: BaseSVC = gs.best_estimator_

    yPredictions: ndarray = model.predict(X=xTest)

    print(accuracy_score(y_true=yTest, y_pred=yPredictions))
