from pathlib import Path
from os.path import abspath
from os import mkdir

def convertRelativePathToAbsolute(path: Path)   ->  Path:
    return Path(abspath(path=path))

def createDirectory(path: Path) ->  None:
    absolutePath: Path = convertRelativePathToAbsolute(path=path)
    mkdir(path=absolutePath)
