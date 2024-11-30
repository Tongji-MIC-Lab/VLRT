import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import blocks
import lib.utils as utils
from lib.config import cfg
from models.basic_model import BasicModel
from models.blocks import (BridgeConnection, LayerStack, PositionwiseFeedForward, ResidualConnection, clone)
from models.multihead_attention import MultiheadedAttention
from models.masking import mask

class AttBasicModel(BasicModel):
    def __init__(self):
        super(AttBasicModel, self).__init__()
        self.ss_prob = 0.0                               # Schedule sampling probability
        self.vocab_size = 10246     # include <BOS>/<EOS>   9363(micdata-en) 13791(micdata-cn)
        self.att_dim = cfg.MODEL.ATT_FEATS_EMBED_DIM \
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0 else cfg.MODEL.ATT_FEATS_DIM

        # word embed
        sequential = [nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.word_embed = nn.Sequential(*sequential)

        # global visual feat embed
        sequential = []
        if cfg.MODEL.GVFEAT_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.GVFEAT_DIM, cfg.MODEL.GVFEAT_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.GVFEAT_EMBED_ACT))
        if cfg.MODEL.DROPOUT_GV_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_GV_EMBED))
        self.gv_feat_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None
        
        # attention spatial feats_embed
        sequential = []
        if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
        self.att_embed_spatial = nn.Sequential(*sequential) if len(sequential) > 0 else None

        # attention lang feats_embed
        sequential = []
        if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_EMBED_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
        self.att_embed_lang = nn.Sequential(*sequential) if len(sequential) > 0 else None
        
        ## attention spatial temporal feats_embed
        sequential = []
        if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_EMBED_DIM*2, cfg.MODEL.ATT_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
        self.att_embed_st = nn.Sequential(*sequential) if len(sequential) > 0 else None
        
        #
        self.dropout_lm  = nn.Dropout(cfg.MODEL.DROPOUT_LM) if cfg.MODEL.DROPOUT_LM > 0 else None
        self.logit = nn.Linear(cfg.MODEL.RNN_SIZE, self.vocab_size)
        self.p_att_feats = nn.Linear(self.att_dim, cfg.MODEL.ATT_HIDDEN_SIZE) \
            if cfg.MODEL.ATT_HIDDEN_SIZE > 0 else None

        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            self.p_att_feats = None
            self.encoder_layers = blocks.create(
                cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
                embed_dim = cfg.MODEL.BILINEAR.DIM, 
                att_type = cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads = cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
                att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
                dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
                layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
            )
        # spatial
        self.attention_spatial = blocks.create(
            cfg.MODEL.BILINEAR.DECODE_BLOCK,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            att_type=cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads=cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            layer_num=cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        self.attention_temporal = blocks.create(
            cfg.MODEL.BILINEAR.ENCODE_BLOCK,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            att_type=cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads=cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim=cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
            att_mid_drop=cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
            dropout=cfg.MODEL.BILINEAR.ENCODE_DROPOUT,
            layer_num=cfg.MODEL.BILINEAR.ENCODE_LAYERS
        )
        # visual && lang
        self.embedding_lang = nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)
        d_model_M1, d_model_M2, H, dout_p, d_model=1024, 1024, 4, 0.1, 1024
        self.self_att_M1 = MultiheadedAttention(d_model_M1, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.self_att_M2 = MultiheadedAttention(d_model_M2, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M1 = MultiheadedAttention(d_model_M1, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M2 = MultiheadedAttention(d_model_M2, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.res_layers_M1 = clone(ResidualConnection(d_model_M1, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_M2, dout_p), 3)
        self.mask = mask
        #self.feed_forward_M1 = PositionwiseFeedForward(d_model_M1, d_ff_M1, dout_p)
        #self.feed_forward_M2 = PositionwiseFeedForward(d_model_M2, d_ff_M2, dout_p)

    def init_hidden(self, batch_size):
        return [Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda())]

    def make_kwargs(self, wt, gv_feat, att_feats, att_mask, p_att_feats, state, **kgs):
        kwargs = kgs
        kwargs[cfg.PARAM.WT] = wt
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.STATE] = state
        return kwargs

    def preprocess(self, **kwargs):
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS_TEMPORAL]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK_TEMPORAL]

        # embed gv_feat
        if self.gv_feat_embed is not None:  ## None
            gv_feat = self.gv_feat_embed(gv_feat)
        
        # embed att_feats
        if self.att_embed_st is not None:
            att_feats = self.att_embed_st(att_feats)

        p_att_feats = self.p_att_feats(att_feats) if self.p_att_feats is not None else None  ## None

        # bilinear
        if cfg.MODEL.BILINEAR.DIM > 0:
            keys, value2s = self.attention.precompute(att_feats, att_feats)
            p_att_feats = torch.cat([keys, value2s], dim=-1)

        return gv_feat, att_feats, att_mask, p_att_feats

    def forward(self, **kwargs): 
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        knowledge = kwargs[cfg.PARAM.KNOWLEDGE]
        att_feats_spatial = kwargs[cfg.PARAM.ATT_FEATS_SPATIAL]
        att_feats_mask_spatial = kwargs[cfg.PARAM.ATT_FEATS_MASK_SPATIAL]
        att_feats_temporal = kwargs[cfg.PARAM.ATT_FEATS_TEMPORAL]
        B, T, dim = att_feats_temporal.size()
        att_feats_temporal = att_feats_temporal.view(B*T, -1).unsqueeze(1)
        att_feats_spatial = torch.cat((att_feats_temporal, att_feats_spatial), dim=1)
        att_feats_spatial=self.att_embed_spatial(att_feats_spatial)
        gv_feat_spatial = torch.ones((1), dtype=torch.long).cuda()
        gv_feat_spatial, att_feats_spatial = self.attention_spatial(gv_feat_spatial, att_feats_spatial, att_feats_mask_spatial, p_att_feats=None, precompute=False)
        att_feats_temporal = gv_feat_spatial.view(B,T,-1)
        gv_feat_temporal = torch.ones((1), dtype=torch.long).cuda()
        att_feats_mask_temporal = kwargs[cfg.PARAM.ATT_FEATS_MASK_TEMPORAL]
        gv_feat_st, att_feats_st = self.attention_temporal(gv_feat_temporal, att_feats_temporal, att_feats_mask_temporal)
        ## visual && lang
        lang_feat = self.embedding_lang(knowledge)
        lang_feat = self.att_embed_lang(torch.mean(lang_feat, dim=2))
        M1, M2= att_feats_st, lang_feat
        M1_mask =self.mask(M1[:, :, 0], None, 1)
        M2_mask =self.mask(M2[:, :, 0], None, 1)
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_att_M1(M1): return self.bi_modal_att_M1(M1, M2, M2, M2_mask)
        def sublayer_att_M2(M2): return self.bi_modal_att_M2(M2, M1, M1, M1_mask)
        # sublayer_ff_M1 = self.feed_forward_M1
        # sublayer_ff_M2 = self.feed_forward_M2
        # Self-Attention
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)
        # Multimodal Attention
        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)
        att_feats_fusion = torch.cat((M1m2, M2m1), dim=-1)
        kwargs[cfg.PARAM.ATT_FEATS_TEMPORAL] = att_feats_fusion
        ##
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).cuda())
        for t in range(seq.size(1)):
            if self.training and t >=1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < self.ss_prob
                if mask.sum() == 0:
                    wt = seq[:,t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = seq[:, t].data.clone()
                    prob_prev = torch.exp(outputs[:, t-1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = seq[:,t].clone()
            if t >= 1 and seq[:, t].max() == 0:
                break
            
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state)
            output, state = self.Forward(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output)

            logit = self.logit(output)
            outputs[:, t] = logit

        return outputs

    def get_logprobs_state(self, **kwargs):
        output, state = self.Forward(**kwargs)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def _expand_state(self, batch_size, beam_size, cur_beam_size, state, selected_beam):
        shape = [int(sh) for sh in state.shape]
        beam = selected_beam
        for _ in shape[2:]:
            beam = beam.unsqueeze(-1)
        beam = beam.unsqueeze(0)
        
        state = torch.gather(
            state.view(*([shape[0], batch_size, cur_beam_size] + shape[2:])), 2,
            beam.expand(*([shape[0], batch_size, beam_size] + shape[2:]))
        )
        state = state.view(*([shape[0], -1, ] + shape[2:]))
        return state

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        knowledge = kwargs[cfg.PARAM.KNOWLEDGE]
        att_feats_spatial = kwargs[cfg.PARAM.ATT_FEATS_SPATIAL]
        att_feats_mask_spatial = kwargs[cfg.PARAM.ATT_FEATS_MASK_SPATIAL]
        att_feats_temporal = kwargs[cfg.PARAM.ATT_FEATS_TEMPORAL]
        B, T, dim = att_feats_temporal.size()
        att_feats_temporal = att_feats_temporal.view(B*T, -1).unsqueeze(1)
        att_feats_spatial = torch.cat((att_feats_temporal, att_feats_spatial), dim=1)
        att_feats_spatial=self.att_embed_spatial(att_feats_spatial)
        gv_feat_spatial = torch.ones((1), dtype=torch.long).cuda()
        gv_feat_spatial, att_feats_spatial = self.attention_spatial(gv_feat_spatial, att_feats_spatial, att_feats_mask_spatial, p_att_feats=None, precompute=False)
        att_feats_temporal = gv_feat_spatial.view(B,T,-1)
        gv_feat_temporal = torch.ones((1), dtype=torch.long).cuda()
        att_feats_mask_temporal = kwargs[cfg.PARAM.ATT_FEATS_MASK_TEMPORAL]
        gv_feat_st, att_feats_st = self.attention_temporal(gv_feat_temporal, att_feats_temporal, att_feats_mask_temporal)
        ## visual && lang
        lang_feat = self.embedding_lang(knowledge)
        lang_feat = self.att_embed_lang(torch.mean(lang_feat, dim=2))
        M1, M2= att_feats_st, lang_feat
        M1_mask =self.mask(M1[:, :, 0], None, 1)
        M2_mask =self.mask(M2[:, :, 0], None, 1)
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_att_M1(M1): return self.bi_modal_att_M1(M1, M2, M2, M2_mask)
        def sublayer_att_M2(M2): return self.bi_modal_att_M2(M2, M1, M1, M1_mask)
        # sublayer_ff_M1 = self.feed_forward_M1
        # sublayer_ff_M2 = self.feed_forward_M2
        # Self-Attention
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)
        # Multimodal Attention
        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)
        att_feats_fusion = torch.cat((M1m2, M2m1), dim=-1)
        kwargs[cfg.PARAM.ATT_FEATS_TEMPORAL] = att_feats_fusion
        ##
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = self.init_hidden(batch_size)
        # seq=0
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        # seq=2
        #wt = Variable(2*torch.ones(batch_size, dtype=torch.long).cuda())

        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            for s in range(len(state)):
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                att_feats = utils.expand_tensor(att_feats, beam_size)
                gv_feat = utils.expand_tensor(gv_feat, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                p_att_feats = utils.expand_tensor(p_att_feats, beam_size)

                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs

    # For the experiments of X-LAN, we use the following beam search code, 
    # which achieves slightly better results but much slower.
    
    #def decode_beam(self, **kwargs):
    #    beam_size = kwargs['BEAM_SIZE']
    #    gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
    #    batch_size = gv_feat.size(0)
    #
    #    sents = Variable(torch.zeros((cfg.MODEL.SEQ_LEN, batch_size), dtype=torch.long).cuda())
    #    logprobs = Variable(torch.zeros(cfg.MODEL.SEQ_LEN, batch_size).cuda())   
    #    self.done_beams = [[] for _ in range(batch_size)]
    #    for n in range(batch_size):
    #        state = self.init_hidden(beam_size)
    #        gv_feat_beam = gv_feat[n:n+1].expand(beam_size, gv_feat.size(1)).contiguous()
    #        att_feats_beam = att_feats[n:n+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
    #        att_mask_beam = att_mask[n:n+1].expand(*((beam_size,)+att_mask.size()[1:]))
    #        p_att_feats_beam = p_att_feats[n:n+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous() if p_att_feats is not None else None
    #
    #        wt = Variable(torch.zeros(beam_size, dtype=torch.long).cuda())
    #        kwargs = self.make_kwargs(wt, gv_feat_beam, att_feats_beam, att_mask_beam, p_att_feats_beam, state, **kwargs)
    #        logprobs_t, state = self.get_logprobs_state(**kwargs)
    #
    #        self.done_beams[n] = self.beam_search(state, logprobs_t, **kwargs)
    #        sents[:, n] = self.done_beams[n][0]['seq'] 
    #        logprobs[:, n] = self.done_beams[n][0]['logps']
    #    return sents.transpose(0, 1), logprobs.transpose(0, 1)

    def decode(self, **kwargs):
        knowledge = kwargs[cfg.PARAM.KNOWLEDGE]
        att_feats_spatial = kwargs[cfg.PARAM.ATT_FEATS_SPATIAL]
        att_feats_mask_spatial = kwargs[cfg.PARAM.ATT_FEATS_MASK_SPATIAL]
        att_feats_temporal = kwargs[cfg.PARAM.ATT_FEATS_TEMPORAL]
        B, T, dim = att_feats_temporal.size()
        att_feats_temporal = att_feats_temporal.view(B*T, -1).unsqueeze(1)
        att_feats_spatial = torch.cat((att_feats_temporal, att_feats_spatial), dim=1)
        att_feats_spatial=self.att_embed_spatial(att_feats_spatial)
        gv_feat_spatial = torch.ones((1), dtype=torch.long).cuda()
        gv_feat_spatial, att_feats_spatial = self.attention_spatial(gv_feat_spatial, att_feats_spatial, att_feats_mask_spatial, p_att_feats=None, precompute=False)
        att_feats_temporal = gv_feat_spatial.view(B,T,-1)
        gv_feat_temporal = torch.ones((1), dtype=torch.long).cuda()
        att_feats_mask_temporal = kwargs[cfg.PARAM.ATT_FEATS_MASK_TEMPORAL]
        gv_feat_st, att_feats_st = self.attention_temporal(gv_feat_temporal, att_feats_temporal, att_feats_mask_temporal)
        ## visual && lang
        lang_feat = self.embedding_lang(knowledge)
        lang_feat = self.att_embed_lang(torch.mean(lang_feat, dim=2))
        M1, M2= att_feats_st, lang_feat
        M1_mask =self.mask(M1[:, :, 0], None, 1)
        M2_mask =self.mask(M2[:, :, 0], None, 1)
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_att_M1(M1): return self.bi_modal_att_M1(M1, M2, M2, M2_mask)
        def sublayer_att_M2(M2): return self.bi_modal_att_M2(M2, M1, M1, M1_mask)
        # sublayer_ff_M1 = self.feed_forward_M1
        # sublayer_ff_M2 = self.feed_forward_M2
        # Self-Attention
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)
        # Multimodal Attention
        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)
        att_feats_fusion = torch.cat((M1m2, M2m1), dim=-1)
        kwargs[cfg.PARAM.ATT_FEATS_TEMPORAL] = att_feats_fusion
        ##
        greedy_decode = kwargs['GREEDY_DECODE']
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        batch_size = gv_feat.size(0)
        state = self.init_hidden(batch_size)

        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs = self.make_kwargs(wt, gv_feat, att_feats, att_mask, p_att_feats, state)
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        return sents, logprobs
