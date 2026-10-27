from typing import Any, TYPE_CHECKING

from .types import Stage
from .dataset_TUM import DatasetTUMRGBDCfg
from .dataset_Replica import DatasetReplicaCfg

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from ..misc.step_tracker import StepTracker
else:
    Dataset = Any
    StepTracker = Any

DatasetCfg = DatasetTUMRGBDCfg | DatasetReplicaCfg


def _dataset_classes():
    from .dataset_TUM import DatasetTUMRGBD
    from .dataset_Replica import DatasetReplica

    return {
        "tum_rgbd": DatasetTUMRGBD,
        "replica": DatasetReplica,
    }


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    from .view_sampler import get_view_sampler

    datasets = _dataset_classes()
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return datasets[cfg.name](cfg, stage, view_sampler)
