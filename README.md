#Representation Learning for point clouds using geometry images

This repo contains the code for training the VAE based geometry image generator method, including hyperparameter tuning with RayTune.

Inside the folder create 'data' and 'results' folders.

In hyperparam.ini you can configure the parameters.

To use ray_tune, install the package, and set up the parameters in the main.py file.

##DATASET

You can download the ModelNet dataset (https://modelnet.cs.princeton.edu/), and then convert it to gim images using https://github.com/sinhayan/learning_geometry_images with Matlab.
Or use the preconverted dataset: https://drive.google.com/drive/folders/1WSO5EysAak148_HufngGzvbRMCr9CjQ6?usp=sharing

Run  ```python main.py --name=my_model``` to train it, you can choose whatever name you prefer.

To test it, run ```python testing.py --name=my_model```. 

Example of the generated point clouds ca be see here, where multiple image sizes were comparised

![plot](./chair_890_test_32_64_128_comparison_labels.png)

The conversion between depth images, point clouds and geometry images can be seen here:

![plot](./depth2pcd2gim_fake.png)

The main file types besides geometry images, are point clouds, RGB images and depth images captured by Pico Zense cameras and ADI Smart Cameras (Courtesy of Analog Devices).

We compared our method to https://github.com/MaciejZamorski/3d-AAE method. Our method performs about 80% of this method concerning Chamfer Distance, but our method is approximately 1.3-1.5 times faster in generating new images.
