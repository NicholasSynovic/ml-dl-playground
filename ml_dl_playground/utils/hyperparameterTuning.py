from typing import Any, List

from sklearn.model_selection import ParameterGrid


def gridSearch(options: dict[str, Any]) -> List[dict[str, Any]]:
    pg: ParameterGrid = ParameterGrid(options)
    return list(pg)
