from .covmatch import covmatch_mtx, covmatch_multi_mtx, covmatch_transfer, select_transfer_features
from .harmony import harmony_mtx, harmony_multi_mtx, harmonize_numpy
try:
    from .mnn import mnn_mtx
except (ImportError, ModuleNotFoundError):
    mnn_mtx = None  # mnnpy._utils Cython ext not built for this Python version
from .scanorama import scanorama_mtx

# backward-compatible aliases for the old CORAL names (pre-refactor scripts)
coral_mtx = covmatch_mtx
select_source_features = select_transfer_features

__all__ = [
    'covmatch_mtx',
    'covmatch_multi_mtx',
    'covmatch_transfer',
    'select_transfer_features',
    'harmony_mtx',
    'harmony_multi_mtx',
    'mnn_mtx',
    'scanorama_mtx',
]
