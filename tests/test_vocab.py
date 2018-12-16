import unittest

from image_captioning.utils.get_vocab import get_vocab


class TestVocab(unittest.TestCase):
    def test_get_vocab(self):
        vocab = get_vocab('coco_2014')
        self.assertIsNotNone(vocab)


if __name__ == "__main__":
    unittest.main(verbosity=2)

