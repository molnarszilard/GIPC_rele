GIPC_rele

mkdir data

mkdir results

Download the ModelNet dataset, convert it to gim images
You can use https://github.com/sinhayan/learning_geometry_images with matlab to do it.

Run main.py to train
In hyperparam.ini you can configure the parameters.
To use ray_tune, install the package, and set up the parameters in the main.py file.

To test it, run testing.py file. 