import torch

from image_captioning.modeling.utils import cat

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched att_features, fc_features
    """
    def __call__(self, batch):
        """

        Args:
            batch (list): a list of data.data is a dict which contains att_feature,
                          fc_feature, caption, caption_len, all_captions

        Returns:
            A dict with the batched data

        """
        att_features,\
        fc_features,\
        captions,\
        cap_lens,\
        all_captions,\
        cocoids = zip(*batch)
        att_features = cat(att_features, dim=0)
        fc_features = cat(fc_features, dim=0)
        captions = torch.stack(captions)
        cap_lens = torch.stack(cap_lens)
        # all ground truth captions for each image
        all_captions = torch.stack(all_captions)

        data = dict()
        data['att_features'] = att_features
        data['fc_features'] = fc_features
        data['captions'] = captions
        data['cap_lens'] = cap_lens
        data['all_captions'] = all_captions
        data['cocoids'] = cocoids

        return data

