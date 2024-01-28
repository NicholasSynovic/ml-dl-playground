from pathlib import Path
from typing import Tuple

from numpy import ndarray
from prepareData import prepare
from train import trainIntelSVC, trainSVC
from warnings import filterwarnings

filterwarnings(action="ignore")


def main() -> None:
    filepath: Path = Path("../../data/iris/bezdekIris.data")

    splits: Tuple[ndarray, ndarray, ndarray, ndarray] = prepare(
        filepath=filepath,
        testSize=0.3,
    )

    trainIntelSVC(data=splits)
    trainSVC(data=splits)


if __name__ == "__main__":
    main()
