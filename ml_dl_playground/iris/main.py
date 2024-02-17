import logging
from pathlib import Path
from time import time
from typing import Any, List, Tuple
from warnings import filterwarnings

from numpy import ndarray
from progress.bar import Bar
from sklearn import metrics as skMetrics
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearnex.svm import SVC as intelSVC

from ml_dl_playground.iris import train
from ml_dl_playground.iris.prepareData import load
from ml_dl_playground.metrics.mlflow import MLFlow
from ml_dl_playground.utils import fs

logging.disable(level=logging.CRITICAL)
filterwarnings(action="ignore")


def _trainingLoop(
    estimator: BaseEstimator,
    gridSearch: List[dict],
    trainValidationSplits: List[Tuple[ndarray, ndarray, ndarray, ndarray]],
    mlf: MLFlow,
) -> None:
    with Bar(f"Training {type(estimator)} models...", max=len(gridSearch)) as bar:
        parameters: dict[str, Any]
        for parameters in gridSearch:
            model: BaseEstimator = estimator.set_params(**parameters)

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
                trainingEndTime: float = time() - trainingStartTime

                predictions: ndarray = model.predict(X=xVal)

                balancedAccuracyScore: float = skMetrics.balanced_accuracy_score(
                    y_true=yVal,
                    y_pred=predictions,
                )
                cohenKappaScore: float = skMetrics.cohen_kappa_score(
                    y1=yVal,
                    y2=predictions,
                )
                linearCohenKappaScore: float = skMetrics.cohen_kappa_score(
                    y1=yVal,
                    y2=predictions,
                    weights="linear",
                )
                quadraticCohenKappaScore: float = skMetrics.cohen_kappa_score(
                    y1=yVal,
                    y2=predictions,
                    weights="quadratic",
                )
                matthewsCorrCoef: float = skMetrics.matthews_corrcoef(
                    y_true=yVal,
                    y_pred=predictions,
                )

                metricsStor: dict[str, Any] = {
                    "training_time": trainingEndTime,
                    "balanced_accuracy_score": balancedAccuracyScore,
                    "cohen_kappa_score": cohenKappaScore,
                    "linear_cohen_kappa_score": linearCohenKappaScore,
                    "quadratic_cohen_kappa_score": quadraticCohenKappaScore,
                    "matthews_corr_coef": matthewsCorrCoef,
                }

                mlf.storeModelInformation(
                    hyperparameters=parameters,
                    tags=modelTags,
                    metrics=metricsStor,
                )

            bar.next()


def main() -> None:
    mlf: MLFlow = MLFlow(experimentName="Iris_SVM")

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
        mlf=mlf,
    )

    _trainingLoop(
        estimator=intelSVC(),
        gridSearch=gridSearch,
        trainValidationSplits=trainValidationSplits,
        mlf=mlf,
    )


if __name__ == "__main__":
    main()
