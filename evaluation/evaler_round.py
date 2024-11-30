import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
from lib.config import cfg
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.masking import mask

def make_masks(feature_stacks, pad_idx):

   masks = mask(feature_stacks[:, :, 0], None, pad_idx)

   return masks

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        eval_annfile
    ):
        super(Evaler, self).__init__()

    def make_kwargs(self, gv_feat, feats_global, feats_spatial, mask_global, mask_spatial, knowledge_fts):
        kwargs={}
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS_GLOBAL] = feats_global
        kwargs[cfg.PARAM.ATT_FEATS_SPATIAL] = feats_spatial
        kwargs[cfg.PARAM.ATT_FEATS_MASK_GLOBAL] = mask_global
        kwargs[cfg.PARAM.ATT_FEATS_MASK_SPATIAL] = mask_spatial
        kwargs[cfg.PARAM.KNOWLEDGE] = knowledge_fts
        ####
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs
        
    def __call__(self, model, rname, test_loader):
        model.eval()
        
        results = []
        predictions = {}
        #test_loader.dataset.update_iterator()
        self.vocab = json.load(open('/root/data1/ltp/codes/video-paragraph-captioning/aa-paper/v2_visual_clip4clip/datas/activitynet/int2word.json'))
        ##
        prediction={}
        with torch.no_grad():
            num=0
            for i, batch in enumerate(test_loader):
                knowledge_fts = torch.LongTensor(batch['knowledge_feature']).cuda()
                img_fts_global = torch.FloatTensor(batch['img_ft_temporal']).cuda()
                img_fts_spatial = torch.FloatTensor(batch['img_ft_spatial']).cuda()
                ft_len = torch.LongTensor(batch['ft_len']).cuda()
                img_fts_global = img_fts_global[:, :max(ft_len)]
                img_fts_spatial = img_fts_spatial[:, :max(ft_len)]
                B1, num1, spatial, num2 = img_fts_spatial.size()
                img_fts_spatial = img_fts_spatial.view(B1 * num1, spatial, num2)
                mask_global = torch.ones((B1, num1), dtype=torch.long).cuda()
                mask_spatial = torch.ones((B1 * num1, spatial+1), dtype=torch.long).cuda()
                gv_feat = torch.zeros(B1, 1).cuda()
                kwargs = self.make_kwargs(gv_feat, img_fts_global, img_fts_spatial, mask_global, mask_spatial, knowledge_fts)
                ##
                if kwargs['BEAM_SIZE'] > 1:
                    #seq, _ = model.module.decode_beam(**kwargs)
                    seq, _ = model.decode_beam(**kwargs)
                else:
                    #seq, _ = model.module.decode(**kwargs)
                    print('beam search is 1')
                    seq, _ = model.decode(**kwargs)
                ##
                sents = utils.decode_sequence(self.vocab, seq.data)
                names=test_loader.dataset.names
                for sent in sents:
                    prediction[names[num]]=sent
                    num+=1
            predict_root = './prediction_results/anet_vrka_v1_rl/predict_' + rname + '.json'
            with open(predict_root, 'w') as result_file:
                 json.dump(prediction, result_file)
        return prediction
