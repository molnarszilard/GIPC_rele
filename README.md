# Representation Learning for point clouds using geometry images

This repo contains the code for training the VAE based geometry image generator method, including hyperparameter tuning with RayTune.

Inside the folder create 'data' and 'results' folders.

Set up the parameters in the main.py file, and in the hyperparam.ini.
Set 'ray' to True, and modify the parameters in the main.py file: (find the line starting with 'if args.ray:')

## SETUP
We recommend using conda environment, for that you can use the environment.yml file. You have to install pytorch separately.

## DATASET

You can download the ModelNet dataset (https://modelnet.cs.princeton.edu/), and then convert it to gim images using https://github.com/sinhayan/learning_geometry_images with Matlab.
Or use the preconverted dataset: https://drive.google.com/drive/folders/1WSO5EysAak148_HufngGzvbRMCr9CjQ6?usp=sharing

Run  ```python main.py --name=my_model``` to train it, you can choose whatever name you prefer.

To test it, run ```python testing.py --name=my_model --checkpoint_dir=results/my_model/```. 

Example of the generated point clouds ca be see here, where multiple image sizes were comparised

![plot](./chair_890_test_32_64_128_comparison_labels.png)

The conversion between depth images, point clouds and geometry images can be seen here:

![plot](./depth2pcd2gim_fake.png)

##  Comparison

We compared our method to https://github.com/MaciejZamorski/3d-AAE method, which uses point clouds.

A table about the results can be seen as the following (at downsampling we copied every second value to the next position, in order to keep the size constant). We used Chamfer Distance as the metric and the runtime (this latter one was quite constant throughout the different tests). At testing the first measured time duration tend be much higher then the rest, because of the initializing phase, so we did not take the first time into account:

| Noise type  | Our | 3d-AAE |
| ------------- | ------------- | ------------- |
| none  | 10.89 | 8.99 |
| Gaussian  | 10.22 | 7.20 |
| Downsampling | 13.36 | 10.24 |
| Average time | 0.022 ms | 0.03 ms |

From the table we can see, that the other method with point clouds performs slightly better as far as the reconstruction goes, however since the main reson of using a VAE architecture is generating new data, this difference could be negligible in different scenarios. Additionally our method is faster in runtime.

## ACKNOWLEDGEMENT

This repo was based on the work of https://github.com/YannDubs/disentangling-vae.

The main file types besides geometry images, are point clouds, RGB images and depth images captured by Pico Zense cameras and ADI Smart Cameras (Courtesy of Analog Devices).