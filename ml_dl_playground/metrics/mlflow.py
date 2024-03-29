from typing import Any

import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities import Experiment
from mlflow.exceptions import RestException


class MLFlow:
    def __init__(
        self,
        experimentName: str,
        experimentTags: dict[str, Any] = {},
        experimentDescription: str = "",
        host: str = "http://localhost",
        port: int = 5000,
        createExperiment: bool = True,
    ) -> None:
        self.experimentName: str = experimentName
        self.experimentTags: dict[str, Any] = experimentTags
        self.experimentDescription: str = experimentDescription
        self.host: str = host
        self.port: int = port

        self.experimentTags["description"] = self.experimentDescription

        self.uri = f"{host}:{port}"

        mlflow.set_tracking_uri(uri=self.uri)

        if createExperiment:
            try:
                mlflow.create_experiment(
                    name=self.experimentName,
                    tags=self.experimentTags,
                )
            except RestException:
                pass

        self.experiment: Experiment = mlflow.set_experiment(
            experiment_name=self.experimentName
        )

        self.experimentID: str = self.experiment.experiment_id

    def storeModelInformation(
        self,
        hyperparameters: dict[str, Any],
        tags: dict[str, Any],
        metrics: dict[str, Any],
        trainingData: Dataset,
        validationData: Dataset,
        testingData: Dataset,
    ) -> None:
        with mlflow.start_run(log_system_metrics=False) as mlfRun:
            mlflow.set_tags(tags=tags)
            mlflow.log_params(params=hyperparameters)
            mlflow.log_metrics(metrics=metrics)
            mlflow.log_input(dataset=trainingData, context="training")
            mlflow.log_input(dataset=validationData, context="validation")
            mlflow.log_input(dataset=testingData, context="testing")
