import json
import os
import sys

from pprint import pprint
import argparse
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

import torch
from torch import nn

class Manager(object):
    def __init__(self, args):
        super(Manager, self).__init__()
        self.args = args
        self.init_dirs()

    @classmethod
    def init(cls):
        # Create a parser
        parser = argparse.ArgumentParser(description="MobileNet by Pytorch")
        parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
        parser.add_argument('--config', default=None, type=str, help='Configuration file')

        # Parse the arguments
        args = parser.parse_args()

        # Parse the configurations from the config json file provided
        config_dir = "./configs/"
        try:
            if args.config is not None:
                with open(config_dir + args.config + '.json', 'r') as config_file:
                    config_args_dict = json.load(config_file)
            else:
                print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
                exit(1)

        except FileNotFoundError:
            print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
            exit(1)
        except json.decoder.JSONDecodeError:
            print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
            exit(1)
        args = edict(config_args_dict)
        pprint(args)
        print("\n")

        return cls(args)

    def init_dirs(self):
        runtime_dir = self.args.runtime.root
        summaries_dir = self.args.runtime.summaries
        checkpoints_dir = self.args.runtime.checkpoints

        experiment_dir = os.path.realpath(
            os.path.join(os.path.dirname(__file__))) + "/../" + runtime_dir
        summary_dir = experiment_dir + summaries_dir + "/"
        checkpoint_dir = experiment_dir + checkpoints_dir + "/"

        dirs = [summary_dir, checkpoint_dir]
        try:
            for dir_ in dirs:
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
            print("Experiment directories created!")
            # return experiment_dir, summary_dir, checkpoint_dir
            self.args.runtime.summary_dir = summary_dir
            self.args.runtime.checkpoint_dir = checkpoint_dir

        except Exception as err:
            print("Creating directories error: {0}".format(err))
            exit(-1)

    def init_model(self, models):
        if self.args.runtime.resume:
            self.load_checkpoint(models)
        else:
            print("Init model weight with random...")
            for item in models:
                item[1].apply(self.weights_init)
            print("Models inited with random.")

    def weights_init(self, model):
        uniform = nn.init.xavier_uniform
        gain = nn.init.calculate_gain('relu')
        if isinstance(model, nn.Conv2d):
            uniform(model.weight.data, gain)
            try:
                uniform(model.bias.data)
            except:
                pass

        elif isinstance(model, nn.Linear):
            uniform(model.weight.data, gain)
            try:
                uniform(model.bias.data)
            except:
                pass

        elif isinstance(model, nn.BatchNorm2d):
            model.weight.data.fill_(1.0)
            model.bias.data.fill_(0)

        elif isinstance(model, nn.BatchNorm1d):
            model.weight.data.fill_(1.0)
            model.bias.data.fill_(0)

    def save_checkpoint(self, models, epoch):
        filename = self.args.runtime.checkpoint_dir + self.args.runtime.checkpoint_file
        state = {
            "epoch": epoch
        }
        for item in models:
            state[item[0]] = item[1].state_dict()
        torch.save(state, filename)

    def load_checkpoint(self, models):
        filename = self.args.runtime.checkpoint_dir + self.args.runtime.checkpoint_file
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.args.optim.last_epoch = checkpoint['epoch']
            self.args.runtime.start_epoch = checkpoint['epoch']
            for item in models:
                item[1].load_state_dict(checkpoint[item[0]])

            print("Checkpoint loaded successfully at (epoch {})\n".format(checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(filename))
            self.args.runtime.resume = False
            self.init_model(models)

    def init_summary(self):
        return SummaryWriter(log_dir=self.args.runtime.summary_dir)
