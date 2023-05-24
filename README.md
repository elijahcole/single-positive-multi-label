# Understanding Label Bias in Single Positive Multi-Label Learning

Code to reproduce the main results in the paper [Understanding Label Bias in Single Positive Multi-Label Learning](https://openreview.net/forum?id=iWiwox99aJ&noteId=LFUBo_ipgS) (ICLR 2023, Tiny Papers Track). 

This repo is an extension of the code for [Multi-Label Learning from Single Positive Labels](https://arxiv.org/abs/2106.09708) (CVPR 2021), which can be found [here](https://github.com/elijahcole/single-positive-multi-label). The primary difference is that this code provides the ability to study the effect of label bias on the COCO dataset. 

## Getting Started

See the `README.md` file in the `data` directory for instructions on downloading and setting up the datasets.

## Training a Model

To train and evaluate a model, run:
```
python train.py
```

## Selecting the Training Procedure
To generate different entries of the main table, modify the following parameters:
1. `dataset`: Which dataset to use.
1. `loss`: Which loss to use.
1. `train_mode`: Whether to (a) train a linear classifier on top of pre-extracted features, (b) train end-to-end, or (c) do (a) followed by (b).
1. `val_set_variant`: Whether to use a clean val set or a validation set where a single positive is observed for each image.
1. `bias_type`: Which sort of biased labels to use (available only for COCO). 

## Hyperparameter Search
As written, `train.py` will run a hyperparameter search over a few different learning rates and batch sizes, save the results for all runs, and report the best run. If desired, modify the code at the bottom of `train.py` to search over different parameter settings. 

**The `linear_init` mode searches over hyperparameters for the fine-tuning phase only.** The hyperparameters for the linear training phase are fixed. In particular, `linear_init_lr` and `linear_init_bsize` are set to the best learning rate and batch size from a `linear_fixed_features` hyperparameter search. 

## Misc
* The `requirements.txt` files was generated using the wonderful tool [pipreqs](https://github.com/bndr/pipreqs).
* Please feel free to get in touch / open an issue if anything is unclear. 
* In this paper we used only those images from NUSWIDE which were still publicly available when we re-crawled the dataset in 2020 using Namhyuk Ahn's [downloader](https://github.com/nmhkahn/NUS-WIDE-downloader). Following the instructions in `data/README.md` should yield the exact subset used for our experiments. 

## Reference  
If you find our work useful in your research please consider citing our papers:  

```latex
@inproceedings{arroyo2023understanding,
  title={Understanding Label Bias in Single Positive Multi-Label Learning},
  author={Arroyo, Julio and 
          Perona, Pietro and 
          Cole, Elijah},
  booktitle={International Conference on Learning Representations, Tiny Papers Track},
  year={2023}
}
@inproceedings{cole2021multi,
  title={Multi-Label Learning from Single Positive Labels},
  author={Cole, Elijah and 
          Mac Aodha, Oisin and 
          Lorieul, Titouan and 
          Perona, Pietro and 
          Morris, Dan and 
          Jojic, Nebojsa},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
