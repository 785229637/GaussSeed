from typing import Any


cfg: Any = None


def get_cfg() -> Any:
    global cfg
    return cfg


def set_cfg(new_cfg: Any) -> None:
    global cfg
    cfg = new_cfg


def get_seed() -> int:
    return cfg.seed
