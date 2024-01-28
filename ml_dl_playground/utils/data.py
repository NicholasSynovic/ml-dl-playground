from pathlib import Path
from typing import List, Tuple

import pandas
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearnex.model_selection import train_test_split


def loadDataFromCSV(dataFilepath: Path, columns: List[str] | None = None) -> DataFrame:
    if columns == None:
        return pandas.read_csv(filepath_or_buffer=dataFilepath)
    else:
        return pandas.read_csv(
            filepath_or_buffer=dataFilepath,
            header=None,
            names=columns,
        )


def prepareClassificationData(
    df: DataFrame,
    classColumnName: str = "class",
    testDataSizeRatio: float = 0.25,
    randomState=42,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Output is in the form of:

    (xTrain, xTest, yTrain, yTest)
    """

    xTrain: ndarray
    xTest: ndarray
    yTrain: ndarray
    yTest: ndarray

    labels: Series = df[classColumnName]
    ndLabels: ndarray = labels.to_numpy(dtype=str)

    values: DataFrame = df.drop(columns="class")
    ndValues: ndarray = values.to_numpy(dtype=float)

    encodedNDLabels: ndarray = LabelEncoder().fit_transform(y=ndLabels)
    normalizedNDValues: ndarray = StandardScaler().fit_transform(X=ndValues)

    xTrain, xTest, yTrain, yTest = train_test_split(
        normalizedNDValues,
        encodedNDLabels,
        test_size=testDataSizeRatio,
        random_state=randomState,
        shuffle=True,
    )

    return (xTrain, xTest, yTrain, yTest)
