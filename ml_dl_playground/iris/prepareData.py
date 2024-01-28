from pathlib import Path
from typing import List, Tuple

from numpy import ndarray
from pandas import DataFrame

from ml_dl_playground.utils import data, fs

COLUMNS: List[str] = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
    "class",
]


def load(
    filepath: Path,
    splits: int = 10,
    randomState: int = 42,
) -> Tuple[List[Tuple[ndarray, ndarray, ndarray, ndarray]], Tuple[ndarray, ndarray]]:
    """
    Output is in the form:

    (
        [
            (
                xTrainSplit_0,
                xValidationSplit_0,
                yTrainSplit_0,
                yValidationSplit_0,
            ),
            (
            ...
            )
        ],
        (
            xTestSplit,
            yTestSplit,
        )
    )
    """

    absoluteFilepath: Path = fs.convertRelativePathToAbsolute(path=filepath)

    df: DataFrame = data.loadDataFromCSV(
        dataFilepath=absoluteFilepath,
        columns=COLUMNS,
    )

    trainTestSplits: Tuple[
        ndarray, ndarray, ndarray, ndarray
    ] = data.prepareClassificationData(
        df=df,
        randomState=randomState,
    )

    trainSplits: Tuple[ndarray, ndarray] = (
        trainTestSplits[0],
        trainTestSplits[2],
    )
    testSplits: Tuple[ndarray, ndarray] = (
        trainTestSplits[1],
        trainTestSplits[3],
    )

    trainValidationSplits: List[
        Tuple[ndarray, ndarray, ndarray, ndarray]
    ] = data.stratifiedKFold(
        x=trainSplits[0],
        y=trainSplits[1],
        splits=splits,
        randomState=randomState,
    )

    return (trainValidationSplits, testSplits)
