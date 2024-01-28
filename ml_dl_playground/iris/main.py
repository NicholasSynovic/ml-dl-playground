from pathlib import Path
from typing import Any, List, Tuple
from warnings import filterwarnings

from numpy import ndarray
from progress.bar import Bar
from sklearn.svm import SVC
from sklearnex.svm import SVC as intelSVC

from ml_dl_playground.iris import train
from ml_dl_playground.iris.prepareData import load
from ml_dl_playground.utils import fs

filterwarnings(action="ignore")


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
    gsSize: int = len(gridSearch)

    with Bar("Training SVC models...", max=gsSize) as bar:
        parameters: dict[str, Any]
        for parameters in gridSearch:
            model: SVC = SVC(**parameters)

            datum: Tuple[ndarray, ndarray, ndarray, ndarray]
            for datum in trainValidationSplits:
                xTrain: ndarray = datum[0]
                yTrain: ndarray = datum[2]

                model.fit(X=xTrain, y=yTrain)

            bar.next()


if __name__ == "__main__":
    main()
