import warnings


def import_pytorch_lightning():
    # filter deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pytorch_lightning as pl

    warnings.filterwarnings("default", category=DeprecationWarning)
    return pl
