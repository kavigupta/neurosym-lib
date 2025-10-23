import logging
import warnings


def import_pytorch_lightning():
    """
    Import pytorch_lightning with warnings suppressed.
    """
    # filter deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # pytorch_lightning (via torch.utils.tensorboard) expects distutils.version to exist.
    # Some environments replace the stdlib distutils module with setuptools._distutils,
    # which does not expose the ``version`` attribute. We patch it in lazily here to
    # avoid import errors without taking a hard dependency on distutils internals.
    import distutils  # pylint: disable=deprecated-module

    if not hasattr(distutils, "version"):
        from distutils import version as _distutils_version  # pylint: disable=deprecated-module

        distutils.version = _distutils_version  # type: ignore[attr-defined]
    import pytorch_lightning as pl

    warnings.filterwarnings("default", category=DeprecationWarning)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*GPU available but not used.*")
    logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("pytorch_lightning.accelerators.cuda")
    logger.setLevel(logging.WARNING)
    return pl
