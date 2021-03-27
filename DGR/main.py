#!/usr/bin/env python3

# Ce code est essentiellement issu du repo github suivant
# https://github.com/kuc2477/pytorch-deep-generative-replay
# Il a été modifié à de nombreux endroits afin de pouvoir traiter notre problème


import argparse
import os.path
import torch
import utils
from train import train
from dgr import Scholar
from models import WGAN, CNN
from data_loader_light2 import ContinualJacquardLoader 

data_path = "../Jacquard_light2/"

parser = argparse.ArgumentParser(
    'PyTorch implementation of Deep Generative Replay'
)


parser.add_argument(
    '--replay-mode', type=str, default='generative-replay',
    choices=['exact-replay', 'generative-replay', 'none'],
)

parser.add_argument('--generator-lambda', type=float, default=10.)
parser.add_argument('--generator-z-size', type=int, default=100)
parser.add_argument('--generator-c-channel-size', type=int, default=64)
parser.add_argument('--generator-g-channel-size', type=int, default=64)
parser.add_argument('--solver-depth', type=int, default=5)
parser.add_argument('--solver-reducing-layers', type=int, default=3)
parser.add_argument('--solver-channel-size', type=int, default=1024)

parser.add_argument('--generator-c-updates-per-g-update', type=int, default=5)
parser.add_argument('--generator-iterations', type=int, default=1000) # 3000
parser.add_argument('--solver-iterations', type=int, default=5000) # 1000
parser.add_argument('--importance-of-new-task', type=float, default=.3)
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-05)
parser.add_argument('--batch-size', type=int, default=32) # 64
parser.add_argument('--test-size', type=int, default=1024)
parser.add_argument('--sample-size', type=int, default=36)

parser.add_argument('--sample-log', action='store_true')
parser.add_argument('--sample-log-interval', type=int, default=300)
parser.add_argument('--image-log-interval', type=int, default=100)
parser.add_argument('--eval-log-interval', type=int, default=50)
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--sample-dir', type=str, default='./samples')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

main_command = parser.add_mutually_exclusive_group(required=False)
main_command.add_argument('--train', action='store_true')
main_command.add_argument('--test', action='store_false', dest='train')


if __name__ == '__main__':
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda
    capacity = args.batch_size * max(
        args.generator_iterations,
        args.solver_iterations
    )

    # dataset
    dataset_config = {
        'size': 224,
        'channels': 4,
        'classes': 5,
    }

    dataset = ContinualJacquardLoader(data_path, args.batch_size)


    # define the models.
    cnn = CNN(
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        classes=dataset_config['classes'],
        depth=args.solver_depth,
        channel_size=args.solver_channel_size,
        reducing_layers=args.solver_reducing_layers,
    )
    wgan = WGAN(
        z_size=args.generator_z_size,
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        c_channel_size=args.generator_c_channel_size,
        g_channel_size=args.generator_g_channel_size,
    )
    label = 'Jacquard-{replay_mode}-r{importance_of_new_task}'.format(
        replay_mode=args.replay_mode,
        importance_of_new_task=(
            1 if args.replay_mode == 'none' else
            args.importance_of_new_task
        ),
    )
    scholar = Scholar(label, generator=wgan, solver=cnn)

    # initialize the model.
    utils.gaussian_intiailize(scholar, std=.02)

    # use cuda if needed
    if cuda:
        scholar.cuda()

    # determine whether we need to train the generator or not.
    train_generator = (
        args.replay_mode == 'generative-replay' or
        args.sample_log
    )

    # run the experiment.
    if True: #args.train:
        train(
            scholar, dataset,
            replay_mode=args.replay_mode,
            generator_lambda=args.generator_lambda,
            generator_iterations=(
                args.generator_iterations if train_generator else 0
            ),
            generator_c_updates_per_g_update=(
                args.generator_c_updates_per_g_update
            ),
            solver_iterations=args.solver_iterations,
            importance_of_new_task=args.importance_of_new_task,
            batch_size=args.batch_size,
            test_size=args.test_size,
            sample_size=args.sample_size,
            lr=args.lr, weight_decay=args.weight_decay,
            beta1=args.beta1, beta2=args.beta2,
            loss_log_interval=args.loss_log_interval,
            eval_log_interval=args.eval_log_interval,
            image_log_interval=args.image_log_interval,
            sample_log_interval=args.sample_log_interval,
            sample_log=args.sample_log,
            sample_dir=args.sample_dir,
            checkpoint_dir=args.checkpoint_dir,
            collate_fn=utils.label_squeezing_collate_fn,
            cuda=cuda
        )
    else:
        path = os.path.join(args.sample_dir, '{}-sample'.format(scholar.name))
        utils.load_checkpoint(scholar, args.checkpoint_dir)
        utils.test_model(scholar.generator, args.sample_size, path)
