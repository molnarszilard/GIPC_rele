import argparse
import logging
import sys
import os
from timeit import default_timer
import numpy as np
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import open3d as o3d

from disvae import init_specific_model
from disvae.models.losses import LOSSES, RECON_DIST
from disvae.models.vae import MODELS
from utils.datasets import DATASETS
from utils.datasets import get_test_datasets
from utils.helpers import set_seed, get_config_section, update_namespace_, FormatterNoDuplicate

TRAIN_LOSSES_LOGFILE = "train_losses.log"
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'])
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')
    training.add_argument('--image_size', type=int, default=default_config['image_size'],
                          help='the size of the image, it is squared, so it would be 32,64,128.')
    training.add_argument('--start_checkpoint', type=int, default=default_config['s_chk'],
                          help='Which epoch would you like to choose. (-1) - start a new model, 0<= - reload another checkpoint')
    training.add_argument('--checkpoint_dir', default=default_config['chk_dir'],
                          help='Path to the testing checkpoint.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

def save_reco(data,images, directory, height):
    images=images*255
    data=data*255
    length = 10
    if len(images)<10:
        length = len(images)
    for i in range(length):
        image = images[i].detach().cpu().numpy().astype(np.uint8)
        while len(image.shape)>3:
            image=image.squeeze(axis=0)
        image = np.moveaxis(image,0,-1)
        path = os.path.join(directory, "gim_test_{}.png".format(i))
        cv2.imwrite(path,image)

        gt = data[i].detach().cpu().numpy().astype(np.uint8)
        while len(gt.shape)>3:
            gt=gt.squeeze(axis=0)
        gt = np.moveaxis(gt,0,-1)
        path = os.path.join(directory, "gim_test_{}_gt.png".format(i))
        cv2.imwrite(path,gt)

        gimnp=np.array(image/255.0)
        gim2flat = np.reshape(gimnp,(height*height,3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gim2flat)
        # pcd = pcd.uniform_down_sample(2)
        path = os.path.join(directory, "gim_test_{}.pcd".format(i)) 
        o3d.io.write_point_cloud(path,pcd)

        gimnp=np.array(gt/255.0)
        gim2flat = np.reshape(gimnp,(height*height,3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gim2flat)
        # pcd = pcd.uniform_down_sample(2)
        path = os.path.join(directory, "gim_test_{}_gt.pcd".format(i)) 
        o3d.io.write_point_cloud(path,pcd)

def test(args, model,device,logger=None, config=None):
    print("Evaluation")
    test_set = get_test_datasets(args.dataset)
    test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batchsize,shuffle=False)
    nr = 0
    start = default_timer()
    model.eval()
    for i, data in enumerate(test_loader, 0):
        nr +=1
        data = data.to(device)
        recon_batch, latent_dist, latent_sample = model(data)
        if i==0:
            save_reco(data,recon_batch, args.checkpoint_dir, args.image_size)
    delta_time = (default_timer() - start)
    print('Finished training after {:.1f} sec.'.format(delta_time))
    print('Processed {} number of batches.'.format(nr))
    print('Time per batch is: {} sec.'.format(delta_time/nr))
    print('Time per image is: {} sec.'.format(delta_time/nr/args.eval_batchsize))

def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    # device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    device = "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    args.img_size = (3,args.image_size,args.image_size)

    model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
    # Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        # if gpus_per_trial > 1:
        #     best_trained_model = nn.DataParallel(best_trained_model)
    
    model_state = torch.load(os.path.join(
        args.checkpoint_dir, "checkpoint"))
    model.load_state_dict(model_state)
    model.to(device)

    test(args=args,model=model,device=device,logger=logger)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
