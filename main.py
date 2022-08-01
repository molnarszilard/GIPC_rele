import argparse
import logging
import sys
import os
from timeit import default_timer
from tqdm import trange
from collections import defaultdict
import numpy as np
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import logging
from torch import optim
import torch
from torch.utils.data import DataLoader, random_split
import cv2
import open3d as o3d
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS, init_specific_model
from utils.datasetloader import DatasetLoader
from utils.helpers import (create_safe_directory, set_seed, 
                           get_config_section, FormatterNoDuplicate)

TRAIN_LOSSES_LOGFILE = "train_losses.log"
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba"]

def get_train_datasets(dataset,batch_size=128):
    train_set = DatasetLoader(root=dataset,train=True)
    return train_set

def get_test_datasets(dataset,batch_size=10):
    test_set = DatasetLoader(root=dataset,train=False)
    return test_set

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
    general.add_argument('--name', type=str,
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
    general.add_argument('--ray', type=bool, default=default_config['ray'],
                         help='Do you want to tune your hyperparameters using ray tune?')
    general.add_argument('--noise', type=str, default='none',
                         help="Do you want to add noise to the input data? Options: none, gauss, dszero, dscopy")

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'])
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

def addnoise(gim):
    noise = (np.random.normal(0, 2, size=gim.shape)).astype(np.float32)/255
    return gim + noise

def downsample(gim):
    for i in range(gim.shape[2]):
        for j in range(gim.shape[3]):
            if (i+j)%2:
                gim[:,:,i,j]=0
    return gim

def downsample_fill(gim):
    for i in range(gim.shape[2]):
        for j in range(gim.shape[3]):
            if (i+j)%2:
                if i%2 and j==0:
                    gim[:,:,i,j]=gim[:,:,i-1,gim.shape[3]-1]
                else:
                    gim[:,:,i,j]=gim[:,:,i,j-1]
    return gim

def train(config=None, args=None):
    model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
    device = "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    # PREPARES DATA
    bs=args.batch_size
    train_set= get_train_datasets(args.dataset)
    test_abs = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(
    train_set, [test_abs, len(train_set) - test_abs])
    train_loader = DataLoader(
        train_subset,
        batch_size=bs,
        shuffle=True,
        num_workers=8)
    val_loader = DataLoader(
        val_subset,
        batch_size=bs,
        shuffle=True,
        num_workers=8)

    # PREPARES MODEL 
    model.to(device)     

    # TRAINS
    lr = args.lr
    if config is not None:
        lr = config['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = get_loss_f(args.loss,config=config,
                        n_data=len(train_loader),
                        device=device,
                        **vars(args))
    start = default_timer()
    
    for epoch in range(args.epochs):
        model.train()
        storer = defaultdict(list)
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch), leave=False,
                    disable=args.no_progress_bar)
        with trange(len(train_loader), **kwargs) as t:
            for _, data in enumerate(train_loader):
                if not args.noise=='none':
                    if args.noise=='gauss':
                        data = addnoise(data)
                    if args.noise=='dszero':
                        data = downsample(data)
                    if args.noise=='dscopy':
                        data = downsample_fill(data)
                data = data.to(device)
                try:
                    recon_batch, latent_dist, latent_sample = model(data)
                    loss,rec_loss, kl_loss, loss_cd = loss_f(data, recon_batch, latent_dist, model.training,
                                    storer, latent_sample=latent_sample)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except ValueError:
                    loss,rec_loss, kl_loss, loss_cd = loss_f.call_optimize(data, model, optimizer, storer)

                iter_loss = loss.item()
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss,rec_loss=rec_loss.item(),kl_loss=kl_loss.item(),loss_cd=loss_cd.item())
                t.update()

        mean_epoch_loss = epoch_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        val_loss_rec = 0.0
        val_loss_kl = 0.0
        val_loss_cd = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                data = data.to(device)
                try:
                    recon_batch, latent_dist, latent_sample = model(data)
                    if args.ray:
                        with tune.checkpoint_dir(epoch) as directory:
                            save_reco(data,recon_batch,directory, args.image_size)
                    else:
                        save_reco(data,recon_batch,args.checkpoint_dir, args.image_size)
                    loss,rec_loss, kl_loss, loss_cd = loss_f(data, recon_batch, latent_dist, not model.training,
                                    storer, latent_sample=latent_sample)
                    total += data.size(0)
                    if loss <10:
                        correct 

                    val_loss += loss.cpu().numpy()
                    val_loss_rec += rec_loss.cpu().numpy()
                    val_loss_kl += kl_loss.cpu().numpy()
                    val_loss_cd += loss_cd.cpu().numpy()
                    val_steps += 1
                except ValueError:
                    loss,rec_loss, kl_loss, loss_cd = loss_f.call_optimize(data, model, optimizer, storer)
        
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs-1:
            if args.ray:
                with tune.checkpoint_dir(epoch) as directory:
                    path_to_model = os.path.join(directory, "checkpoint")
                    torch.save(model.state_dict(), path_to_model)
            else:
                path_to_model = os.path.join(args.checkpoint_dir, "checkpoint")
                torch.save(model.state_dict(), path_to_model)
        if args.ray:
            tune.report(loss=(val_loss / val_steps), rec_loss = (val_loss_rec/val_steps), kl_loss = (val_loss_kl/val_steps), loss_cd = (val_loss_cd/val_steps))
        else:
            print("Losses in epoch {} are:".format(epoch))
            print(  f'(LOSS: {val_loss/ val_steps: .4f}) '
                    f'(CD: {val_loss_cd/ val_steps: .4f}) '
                    f'(KL: {val_loss_kl/ val_steps: .4f}) '
                    f'(RL: {val_loss_rec/ val_steps: .4f}) '
                    )
    delta_time = (default_timer() - start) / 60
    test(args,model,device,logger=None,config=None)

def test(args, model,device,logger=None, config=None):
    print("Evaluation")
    test_set = get_test_datasets(args.dataset)
    test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batchsize,shuffle=False)
    loss_f = get_loss_f(args.loss,config=config,
                        n_data=len(test_loader),
                        device=device,
                        **vars(args))
    nr = 0
    start = default_timer()
    model.eval()
    cd_losses = 0.0
    for i, data in enumerate(test_loader, 0):
        nr +=1
        data = data.to(device)
        recon_batch, latent_dist, latent_sample = model(data)
        loss,rec_loss, kl_loss, loss_cd = loss_f(data, recon_batch, latent_dist, not model.training,
                                storer=None, latent_sample=latent_sample)
        cd_losses += loss_cd
        if i==0:
            save_reco(data,recon_batch, args.checkpoint_dir, args.image_size)
    delta_time = (default_timer() - start)
    print('Finished testing after {:.1f} sec.'.format(delta_time))
    print('Processed {} number of images.'.format(nr))
    print('Time per image is: {} sec.'.format(delta_time/nr))
    print('CD loss per image: {}.'.format(cd_losses/nr))
    return cd_losses

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
    exp_dir = os.path.join(args.checkpoint_dir, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    device = "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda:0"
    args.img_size = (3,args.image_size,args.image_size)
    create_safe_directory(exp_dir, logger=logger)
    if args.ray:
        config = {
            'betaB_finC': tune.grid_search([1,10,50,100,150,200,300,400,500,1000]),
            'betaB_G': tune.grid_search([1,10,50,100,150,1000]),
            'lr': tune.grid_search([1e-4,5e-4,1e-3]),
        }
        gpus_per_trial = 2
        num_samples = 1
        scheduler = ASHAScheduler(
            metric="loss_cd",
            mode="min",
            max_t=args.epochs,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            metric_columns=["loss","rec_loss", "kl_loss","loss_cd","training_iteration"])
        result = tune.run(
            partial(train,args=args),
            name=args.name,
            resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=None,
            progress_reporter=reporter,
            )       
        best_trial = result.get_best_trial("loss_cd", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation reco error: {}".format(
            best_trial.last_result["rec_loss"]))
        print("Best trial final validation kl divergence: {}".format(
            best_trial.last_result["kl_loss"]))
        print("Best trial final validation cd: {}".format(
            best_trial.last_result["loss_cd"]))

        best_trained_model = init_specific_model(args.model_type, args.img_size, args.latent_dim)      
        best_checkpoint_dir = best_trial.checkpoint.value
        args.checkpoint_dir = best_checkpoint_dir
        model_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        best_trained_model.to(device)

        test_acc = test(args=args,model=best_trained_model,device=device,logger=logger, config=best_trial.config)
        print("Best trial test set accuracy: {}".format(test_acc))
    else:
        args.checkpoint_dir=exp_dir
        train(None,args=args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
