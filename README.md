## Description

This repo contains the implementation of image super resolution models **SRResNet** (**S**uper **R**esolution **ResNet**) and SRGAN (**S**uper **R**esolution **GAN**) from scratch using DIV2K, Flickr2K datasets for training and Set5, Set14 datasets for testing. Click [here](https://arxiv.org/abs/1609.04802) to find the paper. Click [here](https://www.kaggle.com/datasets/sharansmenon/div2k) to get DIV2K dataset on kaggle and download it. Click [here](https://www.kaggle.com/datasets/daehoyang/flickr2k) to get Flickr2K dataset on kaggle and download it. Testing datasets can both be downloaded from [here](https://github.com/jbhuang0604/SelfExSR).

The figure below is taken from the paper linked above and represents the architecture of SRGAN.

<p align="center">
<img src="images/architecture.png" width="700"/>
</p>

## Directory Structure and Usage
```
.
├── README.md
│
├── data/                            # dataset directory 
│   ├── train/                       # training datasets; separate dir for each (preferred, but not necessary)
│   │   ├── dataset_1
│   │   ├── dataset_2
│   │   └── ...
│   │
│   └── test/                        # testing datasets; separate dir for each
│
├── images/                          # directory containing images used in this repo
│
├── loss/                            
│   └── loss.py                      # includes loss functions 
│
├── utils/                           
│   ├── dataloader.py                # dataset and batch sampler class
│   ├── model.py                     # includes layers and and model arch code
│   └── utils.py                     # helper functions
│
├── SRGAN_train.py                   # training loop to train SRGAN model
├── SRResNet_train.py                # training loop to train SRResNet model
├── check_image_mode.py              # to check if all images are in RGB mode
├── test.py                          # code to evaluate and get results
└── vgg19_save.py                    # to save pretrained vgg19 model
```

## Installation

Create a python virtual environment / conda environment and install:
```
# first install CUDA 11.8 (only if you have NVIDIA GPU in your machine and compatible NVIDIA driver) 
# install pytorch with CUDA support (refer official docs)

python == 3.10.11
torch == 2.3.1+cu118 
torchvision == 0.18.1+cu118
```

## Steps To Run

1. Clone the project into your machine, and activate the environment in the project dir to run the code.
```
git clone https://github.com/VedantG-02/pytorch-ImageBoost
```
2. Download the dataset from the links provided above and place them in the ```data/train/``` or ```data/test/``` as shown in the above directory structure. Only use the scale factor 4 images in the ```test/``` sets, as the model here upscales 4x the low-res (LR) images.
3. (Optional) To check the count of images in the testing dataset(s) whose mode isn't RGB, run the following command:
```
python check_image_mode.py
```
4. To save the VGG19 pretrained model in order to carry out experiments using VGG Loss (and MSE Loss) as done by authors, run the following command:
```
python vgg19_save.py
```
5. To train SRResNet model (SRGAN uses pretrained SRResNet to initialize its generator), run the following command with appropriate arguments:
```sh
# args provided are the ones with which I trained the models
python SRResNet_train.py --epochs 100 --lr 0.0001 --batch_size 16 --lr_img_size 24 --sf 2 --loss_fn 'MSE'
```
6. To train SRGAN model (with SRResNet MSE as pretrained model), run the following commands (uses different loss functions):
```sh
# MSE Loss
python SRGAN_train.py --from_scratch 'n' --epochs 100 --lr 0.0001 --batch_size 16 --lr_img_size 24 --sf 2 --loss_fn 'MSE'

# VGG Loss
python SRGAN_train.py --from_scratch 'n' --epochs 100 --lr 0.0001 --batch_size 16 --lr_img_size 24 --sf 2 --loss_fn 'VGG' 
--i 2 --j 2
```
7. To evaluate the model on the test datasets and obtain PSNR values as well as generated images, run the following command:
```sh
# testing on Set14 dataset
python test.py --gen_name [ replace_your_gen_name_in_str ] --test_set 'Set5'

# testing on Set5 dataset
python test.py --gen_name [ replace_your_gen_name_in_str ] --test_set 'Set14'
```

## Results
<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="images/original_HR_image.png" alt="Image 1" width="200"/>
        <br>
        <em>Original Image</em>
      </td>
      <td align="center">
        <img src="images/result_SRGAN_MSE.png" alt="Image 2" width="200"/>
        <br>
        <em>SRGAN output</em>
      </td>
      <td align="center">
        <img src="images/result_SRResNet_MSE.png" alt="Image 3" width="200"/>
        <br>
        <em>SRResNet output</em>
      </td>
    </tr>
  </table>
</div>
