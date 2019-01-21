# Image Captioning

This is an image captioning codebase in Pytorch.

main features:
   - instead of including the convnet in the model, we use preprocessed features.
   - use rest101
   - iteration based(this means you need to specify the total iterations)
   - use [karpathy's train-val-test split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
## Requirements:
- pytorch 1.0
- torchvison
- pillow
- lmdb
- tenosrboardX
- yacs

## preprocess 
1. You need modify the dataset settings in `image_captioning/config/paths_catalog.py` or `image_captioning/config/paths_catalog_lmdb.py`
2. run `python setup.py develop` to make sure the imports
3. run `python tools/build_vocab.py` to build vocabulary of the dataset
4. 
    - if you want to use `lmdb` to store the features, make sure `_C.PATHS_CATALOG_PATH=path/to/paths_catalog_lmdb.py`,
and then run `pthon tools/create_lmdb_files.py DATASET.TRAIN coco_2014_train DATASET.VAL cooc_2014_val DATASET.TEST coco_2014_test`
or `python tools/create_lmdb_files.py --config-file path/to/config_file.yaml`(the config file should contain DATASET.TRAIN, DATASET.VAL, DATASET.TEST three options)
    - if you want to store features in a single file, make sure `_C.PATHS_CATALOG_PATH=path/to/paths_catalogb.py`,
and then run `pthon tools/create_input_files.py DATASET.TRAIN coco_2014_train DATASET.VAL cooc_2014_val DATASET.TEST coco_2014_test`
or `python tools/create_input_files.py --config-file path/to/config_file.yaml`(the config file should contain DATASET.TRAIN, DATASET.VAL, DATASET.TEST three options)
5. After generating features, you can run `python tools/train_net.py --config-file path/to/confi_file.yaml` to train the network.


### TODO:
- [] raw image captioning
- [] bottom-up features support

### Acknowledgements
Thanks to [ruotianluo](https://github.com/ruotianluo/self-critical.pytorch) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)