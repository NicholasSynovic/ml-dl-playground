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
    randomState: int = 42,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    absoluteFilepath: Path = fs.convertRelativePathToAbsolute(path=filepath)
    df: DataFrame = data.loadDataFromCSV(
        dataFilepath=absoluteFilepath,
        columns=COLUMNS,
    )
    splits: Tuple[ndarray, ndarray, ndarray, ndarray] = data.prepareClassificationData(
        df=df,
        randomState=randomState,
    )
    return splits
