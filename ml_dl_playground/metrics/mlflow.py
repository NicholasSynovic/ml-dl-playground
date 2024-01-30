from typing import Any

import mlflow
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

    def storeModelInformation(self, hyperparameters: dict[str, Any]) -> None:
        with mlflow.start_run(log_system_metrics=True) as mlfRun:
            mlflow.log_params(params=hyperparameters)
