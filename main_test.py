import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler_round import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
import readers.caption_data as dataset
import framework.run_utils
import framework.logbase
import torch.utils.data as data
from models.masking import mask
##
class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_network()
        self.evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )
        self.path_cfg = framework.run_utils.gen_common_pathcfg('./paths/path_anet.json', is_train=False)
        self.tst_data = dataset.CaptionDataset(self.path_cfg.name_file['tst'],self.path_cfg.ft_root_val, self.path_cfg.cap_file, self.path_cfg.word2int_file, self.path_cfg.int2word_file, self.path_cfg.knowledge_file['tst'], 150, is_train=False, _logger=None)
        self.test_loader = data.DataLoader(self.tst_data, batch_size=20, shuffle=False, num_workers=4)

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_network(self):
        #model = models.create('XLAN')
        model = models.create(cfg.MODEL.TYPE)
        ## load model on single gpu
        self.model=model.cuda()
        ## load model on multiple gpu
        #self.model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
                        
           
                   
    def eval(self, epoch):
        epoch0=15
        for i in range(50):
            print('Now is in the round: [%d]' %(i+epoch0))
            epoch=epoch0 + i*1
            ## for multiple gpus
            #pretrained_dict=torch.load(self.snapshot_path("caption_model", epoch))
            #net_dict=self.model.state_dict()
            #for key, value in pretrained_dict.items():
               #print(key)
            #   net_dict[key[7:]]=value
            #   net_dict.update(net_dict)
            #   self.model.load_state_dict(net_dict)
            ## for single gpu
            self.model.load_state_dict(torch.load(self.snapshot_path("caption_model", epoch),map_location=lambda storage, loc: storage))
            res = self.evaler(self.model, 'test_' + str(epoch), self.test_loader)

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'anet_vrka_v1')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument("--folder", dest='folder', default='./experiments/configs_rl', type=str)
    parser.add_argument("--resume", type=int, default=-1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    tester = Tester(args)
    tester.eval(args.resume)