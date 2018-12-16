import json
import os

from image_captioning.utils.miscellaneous import mkdir


def coco_eval(preds, dataset_name):
    import sys
    sys.path.append('coco-caption')
    ann_file = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    mkdir('eval_results')
    cache_path = os.path.join('eval_results/'+dataset_name+'_results.json')

    coco = COCO(ann_file)
    valids = coco.getImgIds()

    preds_filt = [p for p in preds if p['image_id'] in valids]
    with open(cache_path, 'w') as f:
        json.dump(preds_filt, f)
    coco_res = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, coco_res)
    cocoEval.params['image_id'] = coco_res.getImgIds()
    cocoEval.evaluate()

    out = dict()
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    return out



