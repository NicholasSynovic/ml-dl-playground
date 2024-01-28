from typing import Any, List, Tuple

from numpy import ndarray
from progress.bar import Bar
from sklearn.svm import SVC
from sklearn.svm._base import BaseSVC
from sklearnex.svm import SVC as intelSVC


def _trainSVC(
    estimator: BaseSVC,
    data: Tuple[ndarray, ndarray, ndarray, ndarray],
    epochs: List[int] = [1, 10, 50, 100, 200, 500, 1000],
    randomState: int = 42,
    splits: int = 10,
) -> None:
    xTrain: ndarray = data[0]
    yTrain: ndarray = data[2]

    parameterGrid: dict[str, List[str | int | float] | int] = {
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "max_iter": epochs,
        "random_state": [randomState],
    }

    gs: List[dict[str, Any]] = _gridSearch(options=parameterGrid)
    dataFolds: List[Tuple[ndarray, ndarray, ndarray, ndarray]] = _stratifiedKFold(
        x=xTrain,
        y=yTrain,
        splits=splits,
        randomState=randomState,
    )

    with Bar(f"Training {type(estimator)} models on data...", max=len(gs)) as bar:
        parameters: dict[str, Any]
        for parameters in gs:
            model: BaseSVC = estimator.set_params(**parameters)

            fold: Tuple[ndarray, ndarray, ndarray, ndarray]
            for fold in dataFolds:
                xTrainFold: ndarray = fold[0]
                xValFold: ndarray = fold[1]
                yTrainFold: ndarray = fold[2]
                yValFold: ndarray = fold[3]

                model.fit(X=xTrainFold, y=yTrainFold)

            bar.next()


def trainSVC(
    data: Tuple[ndarray, ndarray, ndarray, ndarray],
    epochs: List[int] = [1, 10, 50, 100, 200, 500, 1000],
    randomState: int = 42,
) -> None:
    _trainSVC(
        estimator=SVC(),
        data=data,
        epochs=epochs,
        randomState=randomState,
    )


def trainIntelSVC(
    data: Tuple[ndarray, ndarray, ndarray, ndarray],
    epochs: List[int] = [1, 10, 50, 100, 200, 500, 1000],
    randomState: int = 42,
) -> None:
    _trainSVC(
        estimator=intelSVC(),
        data=data,
        epochs=epochs,
        randomState=randomState,
    )
