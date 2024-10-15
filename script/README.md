# Quantitative evaluation for comparing motion magnification methods

Many motion magnification methods train their models using the training data proposed by ["Oh, Tae-Hyun, et al., "Learning-based video motion magnification"](https://arxiv.org/abs/1804.02684), ECCV, 2018", but the evaluation data for quantitative assessment presented in that paper has not been made publicly available.

Therefore, we release the evaluation dataset for quantitative comparison of motion magnification methods, strictly following the methods presented in that paper.

## Evaluation dataset information for traditional (generic) motion magnification methods

The structure of the traditional (generic) evaluation dataset is as follows:

    │  Traditional (generic) evaluation dataset
    │   ├── subpixel test dataset
    │   │   ├── mode00: 0.040 (pixel)
    │   │   ├── mode01: 0.050 (pixel)
    │   │   ⋮
    │   │   ├── mode14: 1.000 (pixel)
    │   ├── noise005 test dataset
    │   │   ├── mode00: 0.010 (level of additive noise)
    │   │   ├── mode01: 0.016 (level of additive noise)
    │   │   ⋮₩
    │   │   ├── mode20: 100.0 (level of additive noise)

The subpixel test measures the network's performance for various magnitudes of small motions. This is directly related to the goals of motion magnification methods. 

The noise005 experiment measures the network's performance under various levels of additive noise when the small motion is 0.05 pixels. This is also an important experiment due to the challenge of distinguishing between small motion and photometric noise.

For each mode, the subpixel test uses different magnitudes of small motions, while the noise005 test employs different levels of additive noise.

## Evaluation code for traditional (generic) motion magnification methods
1. Download the generic_evaluation_data.zip file from [this dataset link](https://drive.google.com/drive/folders/1jB2aCfOlQGgAVAzv9lsMDfWlzEIHbYy0) and unzip it.

2. Enter the following command. <--is_single_gpu_trained> command is optional.
    ```
    python script/generic_eval.py --mode subpixel --data_path "Path of generic evaluation data" --checkpoint_path "Path of a pretrained model" --is_single_gpu_trained 
    python script/generic_eval.py --mode noise005 --data_path "Path of generic evaluation data" --checkpoint_path "Path of a pretrained model" --is_single_gpu_trained 
    ```


