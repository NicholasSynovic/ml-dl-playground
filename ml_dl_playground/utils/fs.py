from os import mkdir
from os.path import abspath
from pathlib import Path


def convertRelativePathToAbsolute(path: Path) -> Path:
    return Path(abspath(path=path))


def createDirectory(path: Path) -> None:
    absolutePath: Path = convertRelativePathToAbsolute(path=path)
    mkdir(path=absolutePath)
