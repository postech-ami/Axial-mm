# Learning-based Axial Video Motion Magnification (ECCV 2024)
### [Project Page](https://axial-momag.github.io/axial-momag/) | [Paper](https://arxiv.org/abs/2312.09551)  |  [Dataset](https://drive.google.com/drive/folders/1jB2aCfOlQGgAVAzv9lsMDfWlzEIHbYy0)  |  [YouTube](https://www.youtube.com/watch?v=rirtanavs34)
This repository contains the official implementation of the ECCV 2024 paper, "Learning-based Axial Video Motion Magnification".

## Acknowledgement
I would like to express my gratitude to my advisor, Tae-Hyun Oh, for his outstanding work, which inspired our introduction of user controllability that amplifies motions at specific angles, building upon his paper "Learning-based Motion Magnification."

Most of the code is based on the [Author-verified Pytorch Reimplementation of Learning-based Video Motion Magnification (ECCV 2018)](https://github.com/postech-ami/Deep-Motion-Mag-Pytorch).

## Highlights
**Our proposed axial motion magnification enables the amplification of motion specific to that particular direction.** 

🌟 By amplifying small motion in a specific direction, users can easily understand the object's movement from the results.

🌟 We've added the directional information to motion magnification, which is crucial for applications like fault detection in rotating machinery and building structure health monitoring.

🌟 We've provided [evaluation datasets](https://arxiv.org/abs/2312.09551) for both axial motion magnification and traditional motion magnification. The provided datasets allow for quantitative comparisons between various motion magnification methods.


## 💪To-Do List

- [x] Inference code
- [x] Training code
- [ ] Axial motion magnification quantitative experiment code
- [ ] Traditional motion magnification quantitative experiment code
- [ ] Code for the experiment measuring physical accuracy of motion magnification methods

## Getting started
This code was developed on Ubuntu 18.04 with Python 3.7.6, CUDA 11.1 and PyTorch 1.8.0, using two NVIDIA TITAN RTX (24GB) GPUs. 
Later versions should work, but have not been tested.

### Environment setup

```
conda create -n dmm_pytorch python=3.7.6
conda activate dmm_pytorch

# pytorch installation
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
pip install numpy==1.21.6
pip install pillow tqdm matplotlib scipy tensorboard opencv-python==4.6.0.66
```

## Training
1. Download the training_data.zip file from [this dataset link](https://drive.google.com/drive/folders/1jB2aCfOlQGgAVAzv9lsMDfWlzEIHbYy0) and unzip it.

2. Enter the following command.
    ```
    python main_dp.py --phase="train" --data_path "Path to the directory where the training data is located"
    ```

## Inference
There are various modes for inference in the motion magnification method. Each mode can branch as follows:

    ├── Inference
    │   ├── Without a temporal Filter
    │   │   ├── Static
    │   │   ├── Dynamic
    │   ├── With a temporal filter   
    │   │   ├── differenceOfIIR
    │   │   ├── butter
    │   │   ├── fir

In "Without a temporal filter", the static mode amplifies small motion based on the first frame, while the dynamic mode amplifies small motion by comparing the current frame to the previous frame.

With a temporal filter, amplification is applied by utilizing the temporal filter. This method effectively amplifies small motions of specific frequencies while reducing noise that may arise in the motion magnification results.

🌟 **We highly recommend using a temporal filter for real videos, as they are likely to contain the photometric noise.** 

    
### For the inference without a temporal filter

1) Obtain the [tilted vibration generator video](https://drive.google.com/drive/folders/1Ql0re87ESWPrrZ_fHKWSMGdj34ddBfUn), which is split into multiple frames. When using a custom video, make sure to split it into multiple frames as well.

2) Then, run the static mode for x-axis magnification. Add "--velocity_mag" for dynamic mode.

        python main_dp.py --checkpoint_path "./model/axial_mm.tar" --phase="play" --vid_dir="Path of the video frames" --alpha_x 10 --alpha_y 0 --theta 0 --is_single_gpu_trained   

🌟 **The amplification levels for the x and y axes can be adjusted by setting theta to 0 and modifying <alpha_x> and <alpha_y>. If you want to amplify only one axis, set either <alpha_x> or <alpha_y> to 0** 

🌟 **If you want to amplify at an arbitrary angle, such as 45 degrees, set one of <alpha_x> or <alpha_y> to 0 and input a value for theta between 0 and 90 degrees.** 

### For the inference with a temporal filter

1) And then run the temporal filter mode with differenceOfIIR and FIR filters for y-axis magnification. This code supports three types of <filter_type>, {"differenceOfIIR", "butter", and "fir"}.
      
       python main_dp.py --phase="play_temporal" --is_single_gpu_trained --checkpoint_path "./model/axial_mm.tar"  --vid_dir="Path of the video frames" --alpha_x 0 --alpha_y 10 --theta 0 --fs 120 --freq 15 25 --filter_type fir 
       python main_dp.py --phase="play_temporal" --is_single_gpu_trained --checkpoint_path "./model/axial_mm.tar"  --vid_dir="Path of the video frames" --alpha_x 0 --alpha_y 10 --theta 0 --fs 120 --freq 0.04 0.4 --filter_type differenceOfIIR 

🌟 **When applying a temporal filter, it is crucial to accurately specify the frame rate <fs> and the frequency band <freq> to ensure optimal performance and effectiveness.** 

🌟 **If you want to amplify at an arbitrary angle, such as 45 degrees, set one of <alpha_x> or <alpha_y> to 0 and input a value for <theta> between 0 and 90 degrees.** 

## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@inproceedings{byung2023learning,
  title = {Learning-based Axial Motion Magnification},
  author={Kwon Byung-Ki and Oh Hyun-Bin and Kim Jun-Seong and Hyunwoo Ha and Tae-Hyun Oh},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
````

## Contact
[Kwon Byung-Ki](https://sites.google.com/view/kwon-byung--ki/%ED%99%88) (byungki.kwon@postech.ac.kr)
