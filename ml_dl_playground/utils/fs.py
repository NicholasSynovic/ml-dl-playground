from os import mkdir
from os.path import abspath
from pathlib import Path
from typing import Any, Tuple

from numpy import ndarray
from skl2onnx import to_onnx
from sklearn.base import BaseEstimator

from ml_dl_playground.utils import fs


def convertRelativePathToAbsolute(path: Path) -> Path:
    return Path(abspath(path=path))


def createDirectory(path: Path) -> None:
    absolutePath: Path = convertRelativePathToAbsolute(path=path)
    mkdir(path=absolutePath)


def saveSKLearnModelToONNX(
    model: BaseEstimator,
    outputPath: Path,
    trainingData: ndarray,
) -> None:
    absoluteOutputPath: Path = fs.convertRelativePathToAbsolute(path=outputPath)

    onnxModel: Tuple[Any, Any] = to_onnx(model=model, X=trainingData[:1])
    with open(absoluteOutputPath, "wb") as onnxFile:
        onnxFile.write(onnxModel.SerializeToString())
        onnxFile.close()
