# Getting the Data

## PASCAL

1. Navigate to the PASCAL data directory:
```
cd /path/to/single-positive/multi-label/data/pascal
```
2. Download the data:
```
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar --output pascal_raw.tar
```
3. Extract the data:
```
tar -xf pascal_raw.tar
```
4. Format the data:
```
python format_pascal.py
```
5. Clean up:
```
rm pascal_raw.tar
```
6. Download the pre-extracted features for PASCAL from [here](https://caltech.box.com/v/single-positive-multi-label) and copy them to `/path/to/single-positive/multi-label/data/pascal`.

## COCO

1. Navigate to the COCO data directory:
```
cd /path/to/single-positive/multi-label/data/coco
```
2. Download the data:
```
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
```
3. Extract the data:
```
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
```
4. Format the data:
```
python format_coco.py
```
5. Clean up:
```
rm coco_train_raw.zip
rm coco_val_raw.zip
```
6. Download the pre-extracted features for COCO from [here](https://caltech.box.com/v/single-positive-multi-label) and copy them to `/path/to/single-positive/multi-label/data/coco`.

## NUSWIDE

Coming soon!

## CUB

Coming soon!

# Generating Observed Labels
The script `generate_observed_labels.py` subsamples the entries of a complete label matrix to generate "observed labels" which simulate single positive labeling. To generate observed labels for each dataset, run:
```
python generate_observed_labels.py --dataset X
```
where `X` is replaced by `pascal`, `coco`, `nuswide`, or `cub`. You will only need to do this once.
