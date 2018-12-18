import torch
import time
import logging

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
        logger = logging.getLogger("image_captioning.collat_batch")
        start = time.time()
        att_features = []
        fc_features = []
        captions = []
        cap_lens = []
        all_captions = []
        cocoids = []
        for data in batch:
            att_features.append(data['att_feature'])
            fc_features.append(data['fc_feature'])
            captions.append(data['caption'])
            cap_lens.append(data['cap_len'])
            all_captions.append(data['all_captions'])
            cocoids.append(data['cocoid'])
        att_features = torch.stack(att_features)
        fc_features = torch.stack(fc_features)
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
        end = time.time() - start
        logger.info("collate batch cost: {}".format(end))

        return data

