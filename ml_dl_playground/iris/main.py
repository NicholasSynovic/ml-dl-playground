from pathlib import Path
from typing import Tuple

from numpy import ndarray
from pandas import DataFrame
from prepareData import prepare
from sklearn.model_selection import GridSearchCV
from sklearnex.svm import SVC
from train import score, trainIntelSVC, trainSVC


def main() -> None:
    filepath: Path = Path("../../data/iris/bezdekIris.data")

    splits: Tuple[ndarray, ndarray, ndarray, ndarray] = prepare(
        filepath=filepath,
        testSize=0.3,
    )

    ts: GridSearchCV = trainIntelSVC(data=splits)

    df: DataFrame = DataFrame(data=ts.cv_results_)

    score(gs=ts, data=splits)


if __name__ == "__main__":
    main()
