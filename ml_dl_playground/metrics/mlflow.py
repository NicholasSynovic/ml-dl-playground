from datetime import datetime
from time import time
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment


class MLFlow:
    def __init__(
        self,
        experimentName: str,
        experimentTags: dict[str, Any] = {},
        experimentDescription: str | None = None,
        host: str = "http://localhost",
        port: int = 5000,
        createExperiment: bool = True,
    ) -> None:
        self.experimentName: str = experimentName
        self.experimentTags: dict[str, Any] = experimentTags
        self.experimentDescription: str | None = experimentDescription
        self.host: str = host
        self.port: int = port

        self.experimentTags["description"] = self.experimentDescription
        self.experimentTags["time"] = time()
        self.experimentTags["datetime_UTC"] = datetime.utcnow().__str__()

        self.uri = f"{host}:{port}"

        self.client: MlflowClient = MlflowClient(tracking_uri=self.uri)

        if createExperiment:
            self.client.create_experiment(
                name=self.experimentName,
                tags=self.experimentTags,
            )

        mlflow.set_tracking_uri(uri=self.uri)
        self.experiment: Experiment = mlflow.set_experiment(
            experiment_name=self.experimentName
        )

        self.experimentID: str = self.experiment.experiment_id

    def storeModelHyperParameters(self, hyperparameters: dict[str, Any]) -> None:
        mlflow.log_params(params=hyperparameters)
