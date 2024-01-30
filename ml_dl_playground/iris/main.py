from pathlib import Path
from typing import Any, List, Tuple
from warnings import filterwarnings

from numpy import ndarray
from progress.bar import Bar
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearnex.svm import SVC as intelSVC

from ml_dl_playground.iris import train
from ml_dl_playground.iris.prepareData import load
from ml_dl_playground.utils import fs

filterwarnings(action="ignore")


def _trainingLoop(
    estimator: BaseEstimator,
    gridSearch: List[dict],
    trainValidationSplits: List[Tuple[ndarray, ndarray, ndarray, ndarray]],
) -> None:
    with Bar(f"Training {type(estimator)} models...", max=len(gridSearch)) as bar:
        parameters: dict[str, Any]
        for parameters in gridSearch:
            model: BaseEstimator = estimator.set_params(**parameters)

            datum: Tuple[ndarray, ndarray, ndarray, ndarray]
            for datum in trainValidationSplits:
                xTrain: ndarray = datum[0]
                yTrain: ndarray = datum[2]

                model.fit(X=xTrain, y=yTrain)

            bar.next()


def main() -> None:
    dataFilepath: Path = Path("../../data/iris/bezdekIris.data")
    absoluteDataFilepath: Path = fs.convertRelativePathToAbsolute(path=dataFilepath)

    data: Tuple[
        List[Tuple[ndarray, ndarray, ndarray, ndarray]],
        Tuple[ndarray, ndarray],
    ] = load(filepath=absoluteDataFilepath)

    trainValidationSplits: List[Tuple[ndarray, ndarray, ndarray, ndarray]] = data[0]
    testSplits: Tuple[ndarray, ndarray] = data[1]

    gridSearch: List[dict[str, Any]] = train.DEFAULT_SVC_PARAMETERS

    _trainingLoop(
        estimator=SVC(),
        gridSearch=gridSearch,
        trainValidationSplits=trainValidationSplits,
    )

    _trainingLoop(
        estimator=intelSVC(),
        gridSearch=gridSearch,
        trainValidationSplits=trainValidationSplits,
    )


if __name__ == "__main__":
    main()
