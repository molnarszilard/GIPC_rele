# Representation Learning for point clouds using geometry images

This repo contains the code for training the VAE based geometry image generator method, including hyperparameter tuning with [RayTune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html).

Inside the folder create 'data' and 'results' folders.

Set up the parameters in the main.py file, and in the hyperparam.ini.
Set 'ray' to True, and modify the parameters in the main.py file: (find the line starting with 'if args.ray:')

## Setup
We recommend using conda environment, for that you can use the environment.yml file. You have to [install pytorch separately](https://pytorch.org/get-started/locally/).

## Dataset

You can download the [ModelNet dataset](https://modelnet.cs.princeton.edu/), and then convert it to gim images using https://github.com/sinhayan/learning_geometry_images with Matlab.

Or use the [preconverted dataset](https://drive.google.com/drive/folders/1WSO5EysAak148_HufngGzvbRMCr9CjQ6?usp=sharing)

The conversion between depth images, point clouds and geometry images can be seen here:

![plot](./depth2pcd2gim_fake.png)

## Training and Testing

Run  ```python main.py --name=my_model``` 

To test it, run ```python testing.py --name=my_model --checkpoint_dir=results/my_model/``` 

Example of the generated point clouds ca be see here, where multiple image sizes were comparised. We present the best performing model after tuning.

![plot](./chair_890_test_32_64_128_comparison_labels.png)

##  Comparison

We compared our method to [Adversarial Autoencoders for Compact Representations of 3D Point Clouds](https://arxiv.org/abs/1811.07605) method, which uses point clouds.

To compare the two methods we have used the Chamfer Distance and runtime.

For comparing the chamfer distances we defined 4 cases:

-without noise (the same data as in the training)

-gaussian noise (with mean=0, and standard deviation = 5cm)

-downsampling with zero (every second point/pixel becomes 0, since both methods require a defined size of input data, we cannot delete points/pixels)

-downsampling with copy (every second point/pixel is equal to the previous value)

| Noise type  | Our | 3d-AAE |
| ------------- | ------------- | ------------- |
| without noise  | 5.05 | 3.30 |
| gaussian  | 5.06 | 10.28 |
| downsampling with zero | 6.61 | 3.50 |
| downsampling with copy | 5.66 | 3.41 |

Concerning the runtime (the test were conducted on A100 cards from Nvidia):

| -  | Our | 3d-AAE |
| ------------- | ------------- | ------------- |
| Average time | 1 ms | 2.5 ms |

## Acknowledgment

This repo was based on the work [Disentangled VAE](https://github.com/YannDubs/disentangling-vae).
