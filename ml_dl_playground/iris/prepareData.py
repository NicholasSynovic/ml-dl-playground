from pathlib import Path
from typing import List, Literal, Tuple

import pandas
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer


def _loadData(filepath: Path) -> DataFrame:
    columns: List[str] = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "class",
    ]
    df: DataFrame = pandas.read_csv(
        filepath_or_buffer=filepath,
        header=None,
        names=columns,
    )
    return df


def prepare(
    filepath: Path,
    norm: Literal["l1", "l2", "max", "none"] = "none",
    testSize: float = 0.25,
    randomState: int = 42,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Output is in the form of:

    (xTrain, xTest, yTrain, yTest)
    """

    xTrain: ndarray
    xTest: ndarray
    yTrain: ndarray
    yTest: ndarray

    df: DataFrame = _loadData(filepath=filepath)

    labels: Series = df["class"]
    ndLabels: ndarray = labels.to_numpy(dtype=str)

    values: DataFrame = df.drop(columns="class")
    ndValues: ndarray = values.to_numpy(dtype=float)

    encodedNDLabels: ndarray = LabelEncoder().fit_transform(y=ndLabels)

    normalizedNDValues: ndarray
    if norm == "none":
        normalizedNDValues = ndValues
    else:
        normalizedNDValues: ndarray = Normalizer(norm=norm).fit_transform(X=ndValues)

    xTrain, xTest, yTrain, yTest = train_test_split(
        normalizedNDValues,
        encodedNDLabels,
        test_size=testSize,
        random_state=randomState,
        shuffle=True,
    )

    return (xTrain, xTest, yTrain, yTest)
