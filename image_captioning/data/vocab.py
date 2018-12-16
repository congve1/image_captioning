
class Vocab(object):
    def __init__(self, word_to_ix, ix_to_word):
        self.word_to_ix = word_to_ix 
        self.ix_to_word = ix_to_word

    def __getitem__(self, index):
        if isinstance(index, str):
            word_index = self.word_to_ix.get(index, 
                                            self.word_to_ix['<unk>'])
            return word_index
        elif isinstance(index, int):
            return self.ix_to_word[index]
        else:
            raise Exception("index must be str or int but got {}".format(type(index)))

    def __len__(self):
        return len(self.word_to_ix)