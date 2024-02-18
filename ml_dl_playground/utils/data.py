from pathlib import Path
from typing import Generator, List, Tuple

import pandas
from mlflow.data.numpy_dataset import NumpyDataset, from_numpy
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearnex.model_selection import train_test_split

from ml_dl_playground.utils import fs


def loadDataFromCSV(dataFilepath: Path, columns: List[str] | None = None) -> DataFrame:
    absDataFilepath: Path = fs.convertRelativePathToAbsolute(path=dataFilepath)

    if columns == None:
        return pandas.read_csv(filepath_or_buffer=absDataFilepath)
    else:
        return pandas.read_csv(
            filepath_or_buffer=absDataFilepath,
            header=None,
            names=columns,
        )


def prepareClassificationData(
    df: DataFrame,
    datasetName: str,
    classColumnName: str = "class",
    testDataSizeRatio: float = 0.25,
    randomState=42,
) -> Tuple[NumpyDataset, NumpyDataset]:
    """
    Output is in the form of:

    Tuple[trainingDataset, testingDataset]
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

    trainingDataset: NumpyDataset = from_numpy(
        features=xTrain,
        targets=yTrain,
        name=f"{datasetName}_training",
    )
    testingDataset: NumpyDataset = from_numpy(
        features=xTest,
        targets=yTest,
        name=f"{datasetName}_testing",
    )

    return (trainingDataset, testingDataset)


def stratifiedKFold(
    trainingData: NumpyDataset,
    splits: int = 10,
    randomState=42,
) -> List[Tuple[NumpyDataset, NumpyDataset]]:
    """
    Returns data in the format:

    List[Tuple[trainingDataset, validationDataset]]
    """

    data: List[Tuple[NumpyDataset, NumpyDataset]] = []

    counter: int = 0
    x: ndarray = trainingData.features
    y: ndarray = trainingData.targets
    datasetName: str = trainingData.name

    skf: StratifiedKFold = StratifiedKFold(
        n_splits=splits,
        shuffle=True,
        random_state=randomState,
    )
    folds: Generator = skf.split(X=x, y=y)

    fold: Tuple[ndarray, ndarray]
    for fold in folds:
        trainingIdx: ndarray = fold[0]
        valIdx: ndarray = fold[1]

        xTrain: ndarray = x[trainingIdx]
        yTrain: ndarray = y[trainingIdx]

        xVal: ndarray = x[valIdx]
        yVal: ndarray = y[valIdx]

        trainingDataset: NumpyDataset = from_numpy(
            features=xTrain,
            targets=yTrain,
            name=f"{datasetName}_split_{counter}",
        )
        validationDataset: NumpyDataset = from_numpy(
            features=xVal,
            targets=yVal,
            name=f"{datasetName}_validation_split_{counter}",
        )

        data.append((trainingDataset, validationDataset))

        counter += 1

    return data
