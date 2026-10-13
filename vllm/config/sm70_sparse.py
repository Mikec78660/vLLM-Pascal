# Pascal port: SM70 sparse-attention policy disabled (QSA indexer/E4M3 needs
# CC>=7.0 tensor cores). The model falls back to dense attention on Pascal.
from dataclasses import dataclass


@dataclass(frozen=True)
class QsaTuning:
    score_tile_mb: int = 64
    cublas_min_rows: int = 512
    cublas_min_score_elements: int = 1024 ** 2
    xqa_page4_min_rows: int = 64


SM70_QSA_TUNING = QsaTuning()


class _Disabled:
    active = False
    reason = "SM70 sparse attention disabled on Pascal"

    def __getattr__(self, name):
        return False

    def value(self, name, default=False):
        return default


_disabled = _Disabled()


def sparse_policy(config=None):
    return _disabled


__all__ = ["QsaTuning", "SM70_QSA_TUNING", "sparse_policy"]
