from time import time
from typing import Any, List, Tuple

import pandas
from numpy import ndarray
from pandas import DataFrame
from progress.bar import Bar
from sklearn import metrics as skMetrics
from sklearn.base import BaseEstimator
from ucimlrepo import dotdict, fetch_ucirepo

from ml_dl_playground.metrics.mlflow import MLFlow
from ml_dl_playground.utils import hyperparameterTuning
from ml_dl_playground.utils.data import prepareClassificationData, stratifiedKFold

DEFAULT_SVC_PARAMETERS: List[dict[str, Any]] = hyperparameterTuning.gridSearch(
    options={
        "C": [0.1, 0.5, 1, 2, 5, 10],
        "kernel": ["poly", "rbf", "sigmoid"],
        "max_iter": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        "random_state": [42],
    }
)


def computeMetrics(
    yTrue: ndarray,
    yPredictions: ndarray,
) -> dict[str, Any]:
    balancedAccuracyScore: float = skMetrics.balanced_accuracy_score(
        y_true=yTrue,
        y_pred=yPredictions,
    )
    cohenKappaScore: float = skMetrics.cohen_kappa_score(
        y1=yTrue,
        y2=yPredictions,
    )
    linearCohenKappaScore: float = skMetrics.cohen_kappa_score(
        y1=yTrue,
        y2=yPredictions,
        weights="linear",
    )
    quadraticCohenKappaScore: float = skMetrics.cohen_kappa_score(
        y1=yTrue,
        y2=yPredictions,
        weights="quadratic",
    )
    matthewsCorrCoef: float = skMetrics.matthews_corrcoef(
        y_true=yTrue,
        y_pred=yPredictions,
    )

    return {
        "balanced_accuracy_score": balancedAccuracyScore,
        "cohen_kappa_score": cohenKappaScore,
        "linear_cohen_kappa_score": linearCohenKappaScore,
        "quadratic_cohen_kappa_score": quadraticCohenKappaScore,
        "matthews_corr_coef": matthewsCorrCoef,
    }


def trainingLoop(
    estimator: BaseEstimator,
    gridSearch: List[dict[str, Any]],
    trainValidationSplits: List[Tuple[ndarray, ndarray, ndarray, ndarray]],
    mlf: MLFlow,
) -> None:
    with Bar(
        f"Training {type(estimator)} models...",
        max=len(gridSearch) * len(trainValidationSplits),
    ) as bar:
        hyperparameters: dict[str, Any]
        for hyperparameters in gridSearch:
            model: BaseEstimator = estimator.set_params(**hyperparameters)

            datum: Tuple[ndarray, ndarray, ndarray, ndarray]
            for datum in trainValidationSplits:
                modelTags: dict[str, Any] = {
                    "model_type": str(type(model)),
                }

                xTrain: ndarray = datum[0]
                yTrain: ndarray = datum[2]

                xVal: ndarray = datum[1]
                yVal: ndarray = datum[3]

                trainingStartTime: float = time()
                model.fit(X=xTrain, y=yTrain)
                trainingEndTime: float = time()
                trainingTime: float = trainingEndTime - trainingStartTime

                predictions: ndarray = model.predict(X=xVal)

                modelMetrics: dict[str, Any] = computeMetrics(
                    yTrue=yVal,
                    yPredictions=predictions,
                )

                modelMetrics["training_time"] = trainingTime

                mlf.storeModelInformation(
                    hyperparameters=hyperparameters,
                    tags=modelTags,
                    metrics=modelMetrics,
                )


def loadData() -> DataFrame:
    irisData: dotdict = fetch_ucirepo(id=53)
    irisDataFeatures: DataFrame = irisData["data"]["features"]
    irisDataTargets: DataFrame = irisData["data"]["targets"]
    return pandas.concat(objs=[irisDataFeatures, irisDataTargets], axis=1)


def main() -> None:
    xTrain: ndarray
    yTrain: ndarray
    xTest: ndarray
    yTest: ndarray

    df: DataFrame = loadData()

    xTrain, xTest, yTrain, yTest = prepareClassificationData(df=df)
    trainingData: List[Tuple[ndarray, ndarray, ndarray, ndarray]] = stratifiedKFold(
        x=xTrain, y=yTrain
    )


if __name__ == "__main__":
    main()
