import pickle

from image_captioning.utils.imports import import_file
from image_captioning.config import cfg


def get_vocab(dataset):
    """
    get the vocab of the dataset
    :param dataset: the name of the dataset
    """
    paths_catalog = import_file(
        'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    data = DatasetCatalog.get(dataset)
    vocab_file = data['vocab_file']
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
