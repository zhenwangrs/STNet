import os
import pdb
import time
import torch
# import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from fast_ctc_decode import beam_search, viterbi_search
from torchaudio.models.decoder._ctc_decoder import ctc_decoder


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        # self.ctc_decoder = ctcdecode.CTCBeamDecoder(self.vocab, beam_width=10, blank_id=blank_id, num_processes=10)
        self.vocab = [' ']
        self.vocab.extend(list(gloss_dict.keys()))
        self.py_ctc_decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            lm=None,
            blank_token=" ",
            sil_token=" ",
            beam_size=4,
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        elif self.search_mode == "fast_beam":
            return self.FastBeamSearch(nn_output, vid_lgt, probs)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    def FastBeamSearch(self, nn_output, vid_lgt, probs=False):
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result = self.py_ctc_decoder(emissions=nn_output, lengths=vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0].tokens
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, dim=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            # length = vid_lgt[batch_idx]
            # length = int(length.item())
            # select_index_list = index_list[batch_idx][:length]
            # grouped = groupby(select_index_list)
            group_result = [x[0] for x in groupby(index_list[batch_idx][:int(vid_lgt[batch_idx].item())])]

            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                filtered.append(1)
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(max_result)])
        return ret_list
