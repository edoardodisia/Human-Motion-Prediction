#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
from NN1.utils.constants import *


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        #self.parser.add_argument('--data_dir', type=str, default='./h3.6m/dataset/', help='path to H36M dataset')
        self.parser.add_argument('--data_dir', type=str, default='./data/h3.6m/dataset/', help='path to H36M dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true',
                                 help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='# layers in linear model')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=5.0e-4)
        self.parser.add_argument('--lr_autoencoder', type=float, default=5.0e-4)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=10, help='observed seq length')

        self.parser.add_argument('--output_n', type=int, default=10, help='future seq length')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=256) # MODIFICA: valore originale "default=16"
        self.parser.add_argument('--test_batch', type=int, default=256) # MODIFICA: valore originale "default=16"
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--filename_ext', type=str, default='')

        self.parser.set_defaults(max_norm=True)
        self.parser.set_defaults(is_load=False)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print('MY DEVICE:', MY_DEVICE)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self._print()
        return self.opt
