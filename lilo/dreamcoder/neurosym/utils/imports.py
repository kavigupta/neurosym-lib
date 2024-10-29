import logging
import warnings


def import_pytorch_lightning():
    """
    Import pytorch_lightning with warnings suppressed.
    """
    # filter deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pytorch_lightning as pl

    warnings.filterwarnings("default", category=DeprecationWarning)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*GPU available but not used.*")
    logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("pytorch_lightning.accelerators.cuda")
    logger.setLevel(logging.WARNING)
    return pl
