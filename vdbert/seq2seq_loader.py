from random import randint, shuffle, choices, choice
from random import random as rand
import math
import json
import torch
import torch.utils.data
import torch.nn.functional as F
from vdbert.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

import numpy as np
import h5py
from tqdm import tqdm
import pickle
from typing import List
import copy


class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical
    HDF file is expected to have a column named "image_id", and another column
    named "features".

    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details
    about HDF structure.

    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split
        image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_hdfpath: str, in_memory: bool = False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory

        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self._split = features_hdf.attrs["split"]
            self._image_id_list = list(features_hdf["image_ids"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self._image_id_list)
            self.boxes = [None] * len(self._image_id_list)
            self.classes = [None] * len(self._image_id_list)
            self.scores = [None] * len(self._image_id_list)

    def __len__(self):
        return len(self._image_id_list)

    def __getitem__(self, image_id: int):
        index = self._image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
                boxes = self.boxes[index]
                single_class = self.classes[index]
                single_score = self.scores[index]

            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    boxes = features_hdf["boxes"][index]
                    single_class = features_hdf["classes"][index]
                    single_score = features_hdf["scores"][index]
                    self.features[index] = image_id_features
                    self.boxes[index] = boxes
                    self.classes[index] = single_class
                    self.scores[index] = single_score

        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]
                boxes = features_hdf["boxes"][index]
                single_class = features_hdf["classes"][index]
                single_score = features_hdf["scores"][index]

        return image_id_features, boxes, single_class, single_score

    def keys(self) -> List[int]:
        return self._image_id_list

    @property
    def split(self):
        return self._split


class VisdialDataset(torch.utils.data.Dataset):
    """ Load image-sentence pairs """

    def __init__(self, file_src, batch_size, tokenizer, use_num_imgs=-1, bi_uni_pipeline=[],
                 s2s_prob=1, bi_prob=0, is_train=True, neg_num=0,
                 max_len=80, inc_gt_rel=False, inc_full_hist=False, max_ans_len=10, max_ques_len=20,
                 max_turn_len=30, just_for_pretrain=False, sub_sample=False):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        self.inc_full_hist = inc_full_hist
        print('Sample seq2seq {}, bidirectional {}, negative num {}'.format(self.s2s_prob, self.bi_prob, neg_num))
        assert (self.s2s_prob + self.bi_prob == 1)
        print('Include full history: {}'.format(inc_full_hist))
        if inc_full_hist:
            max_len = 250
        if is_train:
            print("[Train] Remove for the text longer than {} tokens".format(max_len))

        if sub_sample:
            print('Only sample 1 turn from the 10 turns!!')

        # read the file into memory
        self.ex_list = []

        def neg_sample(id, length=100):
            assert 100 > id >= 0
            search_range = list(range(0, id)) + list(range(id + 1, length))
            return choice(search_range)

        with open(file_src, "r", encoding='utf-8') as f_src:
            # raw inputs are given
            data = json.load(f_src)['data']
            counter = 0
            dialogs = data['dialogs']
            questions = data['questions']
            answers = data['answers']
            item_len = []
            long_cnt = 0
            for dialog in tqdm(dialogs):
                if use_num_imgs == -1 or counter < use_num_imgs:
                    img_pth = dialog['image_id']

                    cap_tokens = tokenizer.tokenize(dialog['caption'])
                    ques_id = [item['question'] for item in dialog['dialog']]
                    ans_id = [item['answer'] for item in dialog['dialog']]
                    ans_opts = [item['answer_options'] for item in dialog['dialog']]
                    gt_id = [item['gt_index'] for item in dialog['dialog']]
                    ques_tokens = [tokenizer.tokenize(questions[id] + '?') for id in ques_id]
                    ans_tokens = [tokenizer.tokenize(answers[id]) for id in ans_id]
                    if neg_num != 0:
                        neg_ans_tokens = [[tokenizer.tokenize(answers[ans_opts[turn_i][neg_sample(id)]])
                                           for _ in range(neg_num)] for turn_i, id in enumerate(gt_id)]
                    assert len(ques_tokens) == len(ans_tokens) == 10

                    if sub_sample:
                        sampled_turn_id = randint(0, 9)

                    for turn_i in range(10):
                        if turn_i == 0:
                            hist_tokens = cap_tokens[:20]
                        elif self.inc_full_hist:
                            prev_ans = ans_tokens[turn_i - 1][:8]
                            prev_ques = ques_tokens[turn_i - 1][:20 - len(prev_ans)]
                            hist_tokens += ['[SEP_0]'] + prev_ques + prev_ans

                            # prev_turn = ques_tokens[turn_i - 1] + ans_tokens[turn_i - 1]
                            # hist_tokens += ['[SEP_0]'] + prev_turn[:max_turn_len]  # new boundary
                            # hist_tokens += ['[SEP_0]'] + ques_tokens[turn_i - 1][:max_ques_len] \
                            #                 + ['[SEP_1]'] + ans_tokens[turn_i - 1][:max_ans_len]   # old add boudary
                        else:
                            hist_tokens = ques_tokens[turn_i - 1] + ans_tokens[turn_i - 1]

                        cur_hist = copy.deepcopy(hist_tokens)

                        cur_len = len(cur_hist + ques_tokens[turn_i] + ans_tokens[turn_i])
                        if is_train:
                            if cur_len > max_len:
                                long_cnt += 1
                                continue

                        if just_for_pretrain:
                            if turn_i == 9:
                                self.ex_list.append(
                                    (img_pth, (cur_hist, ques_tokens[turn_i], ans_tokens[turn_i], 1.0)))
                                item_len.append(cur_len)
                        else:
                            if sub_sample:
                                if turn_i == sampled_turn_id:
                                    item_len.append(cur_len)
                                    self.ex_list.append(
                                        (img_pth, (cur_hist, ques_tokens[turn_i], ans_tokens[turn_i], 1.0)))
                                    if neg_num != 0:
                                        for idx in range(neg_num):
                                            self.ex_list.append(
                                                (img_pth, (cur_hist, ques_tokens[turn_i], neg_ans_tokens[turn_i][idx], 0.0)))
                                    break
                            else:
                                item_len.append(cur_len)
                                self.ex_list.append(
                                    (img_pth, (cur_hist, ques_tokens[turn_i], ans_tokens[turn_i], 1.0)))
                                if neg_num != 0:
                                    for idx in range(neg_num):
                                        self.ex_list.append(
                                            (
                                            img_pth, (cur_hist, ques_tokens[turn_i], neg_ans_tokens[turn_i][idx], 0.0)))

                    counter += 1
            print("\nVisdial Img Num: %d with %d examples, Avg len: %.2f, Max len: %d, Min len: %d" %
                  (counter, len(self.ex_list), np.mean(item_len), max(item_len), min(item_len)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        assert len(self.bi_uni_pipeline) == 2
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


def get_gt_rel_dict(fname, gt_key='gt_relevance'):
    gt_rel_dict = {}
    gt_rel_data = json.load(open(fname))
    for item in gt_rel_data:
        image_id = item['image_id']
        round_id = item['round_id']
        gt_relevance = item[gt_key]
        # each image only at most has one turn having dense annotation
        if image_id not in gt_rel_dict:
            gt_rel_dict[image_id] = (round_id, gt_relevance)
    return gt_rel_dict


class VisdialDatasetRelRankLoss(torch.utils.data.Dataset):
    """ Load image-sentence pairs """

    def __init__(self, train_file_src, val_file_src, train_rel_file, val_rel_file, batch_size, tokenizer,
                 use_num_imgs=-1,
                 bi_uni_pipeline=[],
                 s2s_prob=1, bi_prob=0, is_train=True, neg_num=0,
                 max_len=80, inc_gt_rel=False, inc_full_hist=False, add_val=False):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        self.inc_full_hist = inc_full_hist
        print('Sample seq2seq {}, bidirectional {}, negative num {}'.format(self.s2s_prob, self.bi_prob, neg_num))
        assert (self.s2s_prob + self.bi_prob == 1)
        assert neg_num == 1, "Always keep the pos:neg=1:1"
        print('Include full history: {}'.format(inc_full_hist))
        if inc_full_hist:
            max_len = 250
        if is_train:
            print("[Train] Remove for the text longer than {} tokens".format(max_len))

        # read the file into memory
        self.ex_list = []
        train_rel_dict = get_gt_rel_dict(train_rel_file, 'relevance')
        val_rel_dict = get_gt_rel_dict(val_rel_file, 'gt_relevance')

        def get_ex_list(fname, inc_full_hist, rel_dict, add_val=False):
            with open(fname, "r", encoding='utf-8') as f_src:
                # raw inputs are given
                data = json.load(f_src)['data']
                counter = 0
                rel_counter = 0
                non_rel_counter = 0
                dialogs = data['dialogs']
                questions = data['questions']
                answers = data['answers']
                item_len = []
                long_cnt = 0
                ex_list = []
                for dialog in tqdm(dialogs):
                    if use_num_imgs == -1 or counter < use_num_imgs:
                        img_pth = dialog['image_id']
                        if img_pth not in rel_dict:
                            continue

                        cap_tokens = tokenizer.tokenize(dialog['caption'])
                        ques_id = [item['question'] for item in dialog['dialog']]
                        ans_id = [item['answer'] for item in dialog['dialog']]
                        ans_opts = [item['answer_options'] for item in dialog['dialog']]
                        ans_opts_tokens = [[tokenizer.tokenize(answers[id]) for id in ans] for ans in ans_opts]

                        gt_id = [item['gt_index'] for item in dialog['dialog']]
                        ques_tokens = [tokenizer.tokenize(questions[id] + '?') for id in ques_id]
                        ans_tokens = [tokenizer.tokenize(answers[id]) for id in ans_id]

                        assert len(ques_tokens) == len(ans_tokens) == 10

                        for turn_i in range(10):
                            if turn_i == 0:
                                hist_tokens = cap_tokens
                            elif inc_full_hist:
                                # hist_tokens += ques_tokens[turn_i - 1] + ans_tokens[turn_i - 1]
                                prev_ans = ans_tokens[turn_i - 1][:8]
                                prev_ques = ques_tokens[turn_i - 1][:20 - len(prev_ans)]
                                hist_tokens += ['[SEP_0]'] + prev_ques + prev_ans
                            else:
                                hist_tokens = ques_tokens[turn_i - 1] + ans_tokens[turn_i - 1]

                            cur_hist = copy.deepcopy(hist_tokens)

                            cur_len = len(cur_hist + ques_tokens[turn_i] + ans_tokens[turn_i])
                            if is_train:
                                if cur_len > max_len:
                                    long_cnt += 1
                                    continue
                            item_len.append(cur_len)
                            batch_num = 30
                            if inc_gt_rel:
                                if img_pth in rel_dict and turn_i + 1 == rel_dict[img_pth][0]:
                                    rel_scores = rel_dict[img_pth][1]
                                    pos_num = len([1 for rs in rel_scores if rs > 0])
                                    if is_train and pos_num > 30:
                                        break
                                    zero_ids = [idx for idx, rs in enumerate(rel_scores) if rs == 0]
                                    shuffle(zero_ids)
                                    neg_id_set = set(zero_ids[:batch_num-pos_num])
                                    assert len(neg_id_set) == batch_num-pos_num
                                        # print("Long relevance scores: %d at image %d and turn %d" % (
                                        # pos_num, img_pth, turn_i + 1))

                                    ans_batch = []
                                    rs_batch = []
                                    for idx, (ans, rs) in enumerate(zip(ans_opts_tokens[turn_i], rel_scores)):
                                        if rs > 0:
                                            ans_batch.append(ans)
                                            rs_batch.append(rs)
                                            # ex_list.append(
                                            #     (img_pth, (cur_hist, ques_tokens[turn_i], ans, rs)))
                                            rel_counter += 1
                                        elif idx in neg_id_set:
                                            ans_batch.append(ans)
                                            rs_batch.append(0.0)
                                            non_rel_counter += 1
                                            # ex_list.append(
                                            #     (img_pth, (cur_hist, ques_tokens[turn_i], ans, 0.0)))
                                    assert len(rs_batch) == len(ans_batch) == batch_num, "rs batch size: %d" % len(rs_batch)
                                    ex_list.append((img_pth, (cur_hist, ques_tokens[turn_i], ans_batch, rs_batch)))
                                    counter += 1
                print("Reading %d non-zero relevance score items, %d negative samples from %s" % (
                       rel_counter, non_rel_counter, fname))
                print("Total loading data: %d, each has %d options" % (len(ex_list), batch_num))
                return ex_list

        val_ex_list = get_ex_list(val_file_src, self.inc_full_hist, val_rel_dict, add_val)
        train_ex_list = get_ex_list(train_file_src, self.inc_full_hist, train_rel_dict)
        self.ex_list = val_ex_list + train_ex_list
        print("In total, loading %d examples" % len(self.ex_list))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        assert len(self.bi_uni_pipeline) == 2
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4TrainVisdial(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, new_segment_ids=False,
                 truncate_config={}, mask_image_regions=False, mode="s2s", vis_mask_prob=0.25,
                 region_bbox_file='', region_det_file_prefix='', image_features_hdfpath='', visdial_v='1.0',
                 pad_hist=False, finetune=False, only_mask_ans=False, float_nsp_label=False, add_boundary=False,
                 only_qa=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids

        self.len_vis_input = truncate_config.get('len_vis_input', None)
        self.max_len_hist_ques = truncate_config.get('max_len_hist_ques', None)
        self.max_len_ans = truncate_config.get('max_len_ans', None)

        self.mask_image_regions = mask_image_regions
        assert mode in ("s2s", "bi")
        self.mode = mode
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix
        self.visdial_v = visdial_v
        self.pad_hist = pad_hist
        self.finetune = finetune
        self.only_mask_ans = only_mask_ans
        self.float_nsp_label = float_nsp_label
        self.add_boundary = add_boundary
        self.only_qa = only_qa
        if self.only_qa:
            print("Only qa,no previous history!!")

        if visdial_v == '0.9' and self.len_vis_input == 100:
            self.id2img = pickle.load(open('caption_dist/id2img.pkl', 'rb'))
        elif self.len_vis_input == 36:
            self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory=False)
        elif self.len_vis_input == 0:
            pass
        else:
            raise NotImplementedError

        if mode == 's2s':
            self.task_idx = 3  # relax projection layer for different tasks [yue: just reserve this, no effects]
        elif mode == 'bi':
            self.task_idx = 0
        else:
            raise NotImplementedError

        self.vis_mask_prob = vis_mask_prob

    def __call__(self, instance):
        img_path, visdial_example = instance
        tokens_a = ['[UNK]'] * self.len_vis_input

        def pad_to_length(tokens, length):
            tokens = tokens[:length]
            if len(tokens) < length:
                tokens += ['[PAD]'] * (length - len(tokens))
            return tokens

        assert isinstance(visdial_example, tuple)
        hist_tokens, ques_tokens, ans_tokens, nsp_label = visdial_example
        if len(ques_tokens) < self.max_len_hist_ques:
            if self.pad_hist:
                hist_tokens = pad_to_length(hist_tokens, self.max_len_hist_ques - len(ques_tokens))
            else:
                hist_tokens = hist_tokens[:self.max_len_hist_ques - len(ques_tokens)]
        else:
            hist_tokens = []
            ques_tokens = ques_tokens[:self.max_len_hist_ques]

        if self.only_qa:
            prev_tokens = ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
        else:
            prev_tokens = hist_tokens + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
        if self.pad_hist:
            assert len(prev_tokens) == self.max_len_hist_ques + 2
        else:
            assert len(prev_tokens) <= self.max_len_hist_ques + 2

        ans_tokens = ans_tokens[:self.max_len_ans]

        tokens_b = prev_tokens + ans_tokens
        assert len(tokens_b) <= self.max_len_hist_ques + 2 + self.max_len_ans

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        assert len(tokens) <= self.max_len

        if self.new_segment_ids:
            if self.mode == 's2s':
                # segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)  # yue, do not new_segment_ids
            elif self.mode == 'bi':
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            # elif self.mode == 'l2r':
            #     segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        if self.only_mask_ans:
            effective_length = len(ans_tokens)
            start_id = len(prev_tokens) + self.len_vis_input + 2
        else:
            effective_length = len(tokens_b)
            start_id = len(tokens_a) + 2

        if self.max_pred == 0:
            input_ids = self.indexer(tokens)
            masked_ids = masked_pos = masked_weights = []
        else:
            if nsp_label == 0:
                assert tokens.count('[SEP_1]') == 1
                end_id = tokens.index('[SEP_1]')
            else:
                end_id = len(tokens)
            input_ids, masked_ids, masked_pos, masked_weights = self.conduct_mask(tokens,
                                                                                  effective_length,
                                                                                  start_id, end_id)
        input_mask, vis_masked_pos = self.get_attn_mask(tokens, len(prev_tokens))
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        img, vis_pe = self.get_butd(img_path)

        if self.float_nsp_label:
            nsp_label = torch.tensor(nsp_label, dtype=torch.float32)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, nsp_label, self.task_idx,
                vis_masked_pos, img, vis_pe)

    def get_attn_mask(self, tokens, prev_tokens_len):
        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

        if self.mode == "bi":
            input_mask[:, :len(tokens)].fill_(1)
        elif self.mode == "s2s":
            if self.only_mask_ans:
                input_mask[:, :self.len_vis_input + 2 + prev_tokens_len].fill_(1)
                second_st, second_end = self.len_vis_input + 2 + prev_tokens_len, len(tokens)
            else:
                input_mask[:, :self.len_vis_input + 2].fill_(1)
                second_st, second_end = self.len_vis_input + 2, len(tokens)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])

        if self.pad_hist:
            padded_pos = [id for id, t in enumerate(tokens) if t == '[PAD]']
            if len(padded_pos) > 0:
                assert len(padded_pos) == padded_pos[-1] - padded_pos[0] + 1
                input_mask[:, padded_pos[0]:padded_pos[-1] + 1].fill_(0)

        if self.mask_image_regions:
            vis_masked_pos = np.random.choice(self.len_vis_input,
                                              int(self.len_vis_input * self.vis_mask_prob),
                                              replace=False) + 1  # +1 for [CLS], always of the same length
        else:
            vis_masked_pos = []

        if self.mask_image_regions:
            input_mask[:, vis_masked_pos].fill_(0)  # block the masked visual feature
        return input_mask, vis_masked_pos

    def conduct_mask(self, tokens, effective_length, start_id, end_id):
        # For masked Language Models
        cand_pos = []
        special_pos = set()

        n_pred = min(self.max_pred, max(
            1, int(round(effective_length * self.mask_prob))))

        # candidate positions of masked tokens
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= start_id) and (tk != '[CLS]') and (tk != '[PAD]') and (i < end_id):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)
        masked_pos = cand_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if self.finetune:
                tokens[pos] = '[MASK]'
                continue
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        assert len(masked_ids) == len(masked_pos) == len(masked_weights) == self.max_pred, \
            "[masked] id: %d, pos: %d, weights: %d" % (len(masked_ids), len(masked_pos), len(masked_weights))

        return input_ids, masked_ids, masked_pos, masked_weights

    def get_butd(self, img_id):
        if self.len_vis_input == 100 and self.visdial_v == '0.9':
            last_3_id_str = '{:0>12d}'.format(img_id)[-3:]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix + '_feat' + last_3_id_str + '.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix + '_cls' + last_3_id_str + '.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img_id_key = self.id2img[img_id]  # example of img_id_key: COCO_train2014_00000000000
                    img = torch.from_numpy(region_feat_f[img_id_key][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id_key][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id_key][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1)  # confident score
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                                F.layer_norm(cls_label, [1601])), dim=-1)  # 1601 hard coded...
        elif self.len_vis_input == 36:
            img, boxes, single_class, single_score = self.hdf_reader[img_id]
            img = torch.Tensor(img)
            boxes = torch.Tensor(boxes)
            single_class = torch.Tensor(single_class)
            single_score = torch.Tensor(single_score)

            w_est = torch.max(boxes[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(boxes[:, [1, 3]]) * 1. + 1e-5
            boxes[:, [0, 2]] /= w_est
            boxes[:, [1, 3]] /= h_est
            rel_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((boxes[:, :4], rel_area.view(-1, 1), single_class.view(-1, 1), single_score.view(-1, 1)),
                               -1)  # 7 dim
        elif self.len_vis_input == 0:
            return [], []
        else:
            raise NotImplementedError
        return img, vis_pe


class Preprocess4TrainVisdialRankLoss(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, new_segment_ids=False,
                 truncate_config={}, mask_image_regions=False, mode="s2s", vis_mask_prob=0.25,
                 region_bbox_file='', region_det_file_prefix='', image_features_hdfpath='', visdial_v='1.0',
                 pad_hist=False, finetune=False, only_mask_ans=False, float_nsp_label=False, add_boundary=False,
                 only_qa=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids

        self.len_vis_input = truncate_config.get('len_vis_input', None)
        self.max_len_hist_ques = truncate_config.get('max_len_hist_ques', None)
        self.max_len_ans = truncate_config.get('max_len_ans', None)

        self.mask_image_regions = mask_image_regions
        assert mode in ("s2s", "bi")
        self.mode = mode
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix
        self.visdial_v = visdial_v
        self.pad_hist = pad_hist
        self.finetune = finetune
        self.only_mask_ans = only_mask_ans
        self.float_nsp_label = float_nsp_label
        self.add_boundary = add_boundary
        self.only_qa = only_qa
        if self.only_qa:
            print("Only qa,no previous history!!")

        if visdial_v == '0.9' and self.len_vis_input == 100:
            self.id2img = pickle.load(open('caption_dist/id2img.pkl', 'rb'))
        elif self.len_vis_input == 36:
            self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory=False)
        elif self.len_vis_input == 0:
            pass
        else:
            raise NotImplementedError

        if mode == 's2s':
            self.task_idx = 3  # relax projection layer for different tasks [yue: just reserve this, no effects]
        elif mode == 'bi':
            self.task_idx = 0
        else:
            raise NotImplementedError

        self.vis_mask_prob = vis_mask_prob

    def __call__(self, instance):
        img_path, visdial_example = instance
        tokens_a = ['[UNK]'] * self.len_vis_input

        def pad_to_length(tokens, length):
            tokens = tokens[:length]
            if len(tokens) < length:
                tokens += ['[PAD]'] * (length - len(tokens))
            return tokens

        assert isinstance(visdial_example, tuple)
        hist_tokens, ques_tokens, ans_tokens_batch, nsp_label_batch = visdial_example
        if len(ques_tokens) < self.max_len_hist_ques:
            if self.pad_hist:
                hist_tokens = pad_to_length(hist_tokens, self.max_len_hist_ques - len(ques_tokens))
            else:
                hist_tokens = hist_tokens[:self.max_len_hist_ques - len(ques_tokens)]
        else:
            hist_tokens = []
            ques_tokens = ques_tokens[:self.max_len_hist_ques]

        if self.only_qa:
            prev_tokens = ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
        else:
            prev_tokens = hist_tokens + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
        if self.pad_hist:
            assert len(prev_tokens) == self.max_len_hist_ques + 2
        else:
            assert len(prev_tokens) <= self.max_len_hist_ques + 2

        example_batch = []
        for ans_tokens, nsp_label in zip(ans_tokens_batch, nsp_label_batch):
            ans_tokens = ans_tokens[:self.max_len_ans]

            tokens_b = prev_tokens + ans_tokens
            assert len(tokens_b) <= self.max_len_hist_ques + 2 + self.max_len_ans

            # Add Special Tokens
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            assert len(tokens) <= self.max_len

            if self.new_segment_ids:
                if self.mode == 's2s':
                    # segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)  # yue, do not new_segment_ids
                elif self.mode == 'bi':
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                # elif self.mode == 'l2r':
                #     segment_ids = [2] * (len(tokens))
            else:
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            if self.only_mask_ans:
                effective_length = len(ans_tokens)
                start_id = len(prev_tokens) + self.len_vis_input + 2
            else:
                effective_length = len(tokens_b)
                start_id = len(tokens_a) + 2

            if self.max_pred == 0:
                input_ids = self.indexer(tokens)
                masked_ids = masked_pos = masked_weights = []
            else:
                input_ids, masked_ids, masked_pos, masked_weights = self.conduct_mask(tokens,
                                                                                      effective_length,
                                                                                      start_id)
            input_mask, vis_masked_pos = self.get_attn_mask(tokens, len(prev_tokens))
            n_pad = self.max_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad)

            img, vis_pe = self.get_butd(img_path)

            if self.float_nsp_label:
                nsp_label = torch.tensor(nsp_label, dtype=torch.float32)
            example = (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, nsp_label, self.task_idx,
                    vis_masked_pos, img, vis_pe)
            example_batch.append(example)
        return example_batch

    def get_attn_mask(self, tokens, prev_tokens_len):
        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

        if self.mode == "bi":
            input_mask[:, :len(tokens)].fill_(1)
        elif self.mode == "s2s":
            if self.only_mask_ans:
                input_mask[:, :self.len_vis_input + 2 + prev_tokens_len].fill_(1)
                second_st, second_end = self.len_vis_input + 2 + prev_tokens_len, len(tokens)
            else:
                input_mask[:, :self.len_vis_input + 2].fill_(1)
                second_st, second_end = self.len_vis_input + 2, len(tokens)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])

        if self.pad_hist:
            padded_pos = [id for id, t in enumerate(tokens) if t == '[PAD]']
            if len(padded_pos) > 0:
                assert len(padded_pos) == padded_pos[-1] - padded_pos[0] + 1
                input_mask[:, padded_pos[0]:padded_pos[-1] + 1].fill_(0)

        if self.mask_image_regions:
            vis_masked_pos = np.random.choice(self.len_vis_input,
                                              int(self.len_vis_input * self.vis_mask_prob),
                                              replace=False) + 1  # +1 for [CLS], always of the same length
        else:
            vis_masked_pos = []

        if self.mask_image_regions:
            input_mask[:, vis_masked_pos].fill_(0)  # block the masked visual feature
        return input_mask, vis_masked_pos

    def conduct_mask(self, tokens, effective_length, start_id):
        # For masked Language Models
        cand_pos = []
        special_pos = set()

        n_pred = min(self.max_pred, max(
            1, int(round(effective_length * self.mask_prob))))

        # candidate positions of masked tokens
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= start_id) and (tk != '[CLS]') and (tk != '[PAD]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)
        masked_pos = cand_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if self.finetune:
                tokens[pos] = '[MASK]'
                continue
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        assert len(masked_ids) == len(masked_pos) == len(masked_weights) == self.max_pred, \
            "[masked] id: %d, pos: %d, weights: %d" % (len(masked_ids), len(masked_pos), len(masked_weights))

        return input_ids, masked_ids, masked_pos, masked_weights

    def get_butd(self, img_id):
        if self.len_vis_input == 100 and self.visdial_v == '0.9':
            last_3_id_str = '{:0>12d}'.format(img_id)[-3:]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix + '_feat' + last_3_id_str + '.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix + '_cls' + last_3_id_str + '.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img_id_key = self.id2img[img_id]  # example of img_id_key: COCO_train2014_00000000000
                    img = torch.from_numpy(region_feat_f[img_id_key][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id_key][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id_key][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1)  # confident score
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                                F.layer_norm(cls_label, [1601])), dim=-1)  # 1601 hard coded...
        elif self.len_vis_input == 36:
            img, boxes, single_class, single_score = self.hdf_reader[img_id]
            img = torch.Tensor(img)
            boxes = torch.Tensor(boxes)
            single_class = torch.Tensor(single_class)
            single_score = torch.Tensor(single_score)

            w_est = torch.max(boxes[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(boxes[:, [1, 3]]) * 1. + 1e-5
            boxes[:, [0, 2]] /= w_est
            boxes[:, [1, 3]] /= h_est
            rel_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((boxes[:, :4], rel_area.view(-1, 1), single_class.view(-1, 1), single_score.view(-1, 1)),
                               -1)  # 7 dim
        elif self.len_vis_input == 0:
            return [], []
        else:
            raise NotImplementedError
        return img, vis_pe


class Preprocess4TestVisdialDisc(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, new_segment_ids=False,
                 truncate_config={}, mode="s2s",
                 region_bbox_file='', region_det_file_prefix='', image_features_hdfpath='', visdial_v='1.0',
                 pad_hist=False, inc_full_hist=False, only_qa=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids

        self.len_vis_input = truncate_config.get('len_vis_input', None)
        self.max_len_hist_ques = truncate_config.get('max_len_hist_ques', None)
        self.max_len_ans = truncate_config.get('max_len_ans', None)

        assert mode in ("bi")
        self.mode = mode
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix
        self.visdial_v = visdial_v
        self.pad_hist = pad_hist
        self.inc_full_hist = inc_full_hist
        self.only_qa = only_qa

        if visdial_v == '0.9' and self.len_vis_input == 100:
            self.id2img = pickle.load(open('caption_dist/id2img.pkl', 'rb'))
        elif self.len_vis_input == 36:
            self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory=True)
        elif self.len_vis_input == 0:
            pass
        else:
            raise NotImplementedError

        if mode == 's2s':
            self.task_idx = 3  # relax projection layer for different tasks [yue: just reserve this, no effects]
        elif mode == 'bi':
            self.task_idx = 0
        else:
            raise NotImplementedError

    def __call__(self, instance):
        img_path, cap_tokens, ques_tokens_turns, ans_tokens_turns, ans_opts_tokens_turns = instance
        tokens_a = ['[UNK]'] * self.len_vis_input

        def pad_to_length(tokens, length):
            tokens = tokens[:length]
            if len(tokens) < length:
                tokens += ['[PAD]'] * (length - len(tokens))
            return tokens

        input_ids_turn = []
        segment_ids_turn = []
        input_mask_turn = []
        turn_num = len(ques_tokens_turns)

        for turn_i in range(turn_num):

            if turn_i == 0:
                hist_tokens = cap_tokens[:20]
            else:
                prev_ans = ans_tokens_turns[turn_i - 1][:8]
                prev_ques = ques_tokens_turns[turn_i - 1][:20 - len(prev_ans)]
                if self.inc_full_hist:
                    hist_tokens += ['[SEP_0]'] + prev_ques + prev_ans
                else:
                    hist_tokens = prev_ques + prev_ans
            cur_hist = copy.deepcopy(hist_tokens)

            ques_tokens = ques_tokens_turns[turn_i]

            if len(ques_tokens) < self.max_len_hist_ques:
                if self.pad_hist:
                    cur_hist = pad_to_length(cur_hist, self.max_len_hist_ques - len(ques_tokens))
                else:
                    cur_hist = cur_hist[:self.max_len_hist_ques - len(ques_tokens)]
            else:
                cur_hist = []
                ques_tokens = ques_tokens[:self.max_len_hist_ques]

            if self.only_qa:
                prev_tokens = ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
            else:
                prev_tokens = cur_hist + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
            if self.pad_hist:
                assert len(prev_tokens) == self.max_len_hist_ques + 2
            else:
                assert len(prev_tokens) <= self.max_len_hist_ques + 2

            input_ids_opt = []
            segment_ids_opt = []
            input_mask_opt = []

            for opt_id, cur_ans_opt in enumerate(ans_opts_tokens_turns[turn_i]):
                # cur_ans_opt = cur_ans_opt[:self.max_len_ans]
                ans_tokens = cur_ans_opt[:self.max_len_ans]

                tokens_b = prev_tokens + ans_tokens
                assert len(tokens_b) <= self.max_len_hist_ques + 2 + self.max_len_ans

                # Add Special Tokens
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
                assert len(tokens) <= self.max_len

                if self.new_segment_ids:
                    if self.mode == 's2s':
                        # segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (
                                len(tokens_b) + 1)  # yue, do not new_segment_ids
                    elif self.mode == 'bi':
                        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                    # elif self.mode == 'l2r':
                    #     segment_ids = [2] * (len(tokens))
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                input_ids = self.indexer(tokens)

                input_mask = self.get_attn_mask(tokens, len(prev_tokens))
                n_pad = self.max_len - len(input_ids)
                input_ids.extend([0] * n_pad)
                segment_ids.extend([0] * n_pad)

                input_ids_opt.append(input_ids)
                segment_ids_opt.append(segment_ids)
                input_mask_opt.append(input_mask)

            input_ids_turn.append(input_ids_opt)
            segment_ids_turn.append(segment_ids_opt)
            input_mask_turn.append(torch.stack(input_mask_opt))

        img, vis_pe = self.get_butd(img_path)
        return (input_ids_turn, segment_ids_turn, input_mask_turn, self.task_idx, img, vis_pe)

    def get_attn_mask(self, tokens, prev_tokens_len):
        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

        if self.mode == "bi":
            input_mask[:, :len(tokens)].fill_(1)
        elif self.mode == "s2s":
            if self.only_mask_ans:
                input_mask[:, :self.len_vis_input + 2 + prev_tokens_len].fill_(1)
                second_st, second_end = self.len_vis_input + 2 + prev_tokens_len, len(tokens)
            else:
                input_mask[:, :self.len_vis_input + 2].fill_(1)
                second_st, second_end = self.len_vis_input + 2, len(tokens)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])

        if self.pad_hist:
            padded_pos = [id for id, t in enumerate(tokens) if t == '[PAD]']
            if len(padded_pos) > 0:
                assert len(padded_pos) == padded_pos[-1] - padded_pos[0] + 1
                input_mask[:, padded_pos[0]:padded_pos[-1] + 1].fill_(0)
        #
        # if self.mask_image_regions:
        #     vis_masked_pos = np.random.choice(self.len_vis_input,
        #                                       int(self.len_vis_input * self.vis_mask_prob),
        #                                       replace=False) + 1  # +1 for [CLS], always of the same length
        # else:
        #     vis_masked_pos = []
        #
        # if self.mask_image_regions:
        #     input_mask[:, vis_masked_pos].fill_(0)  # block the masked visual feature
        return input_mask

    def conduct_mask(self, tokens, effective_length, start_id):
        # For masked Language Models
        cand_pos = []
        special_pos = set()

        n_pred = min(self.max_pred, max(
            1, int(round(effective_length * self.mask_prob))))

        # candidate positions of masked tokens
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= start_id) and (tk != '[CLS]') and (tk != '[PAD]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)
        masked_pos = cand_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if self.finetune:
                tokens[pos] = '[MASK]'
                continue
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        assert len(masked_ids) == len(masked_pos) == len(masked_weights) == self.max_pred, \
            "[masked] id: %d, pos: %d, weights: %d" % (len(masked_ids), len(masked_pos), len(masked_weights))

        return input_ids, masked_ids, masked_pos, masked_weights

    def get_butd(self, img_id):
        if self.len_vis_input == 100 and self.visdial_v == '0.9':
            last_3_id_str = '{:0>12d}'.format(img_id)[-3:]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix + '_feat' + last_3_id_str + '.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix + '_cls' + last_3_id_str + '.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img_id_key = self.id2img[img_id]  # example of img_id_key: COCO_train2014_00000000000
                    img = torch.from_numpy(region_feat_f[img_id_key][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id_key][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id_key][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1)  # confident score
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                                F.layer_norm(cls_label, [1601])), dim=-1)  # 1601 hard coded...
        elif self.len_vis_input == 36:
            img, boxes, single_class, single_score = self.hdf_reader[img_id]
            img = torch.Tensor(img)
            boxes = torch.Tensor(boxes)
            single_class = torch.Tensor(single_class)
            single_score = torch.Tensor(single_score)

            w_est = torch.max(boxes[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(boxes[:, [1, 3]]) * 1. + 1e-5
            boxes[:, [0, 2]] /= w_est
            boxes[:, [1, 3]] /= h_est
            rel_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((boxes[:, :4], rel_area.view(-1, 1), single_class.view(-1, 1), single_score.view(-1, 1)),
                               -1)  # 7 dim
        elif self.len_vis_input == 0:
            return [], []
        else:
            raise NotImplementedError
        return img, vis_pe


class Preprocess4TestVisdialDiscTest(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, new_segment_ids=False,
                 truncate_config={}, mode="s2s",
                 region_bbox_file='', region_det_file_prefix='', image_features_hdfpath='', visdial_v='1.0',
                 pad_hist=False, inc_full_hist=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids

        self.len_vis_input = truncate_config.get('len_vis_input', None)
        self.max_len_hist_ques = truncate_config.get('max_len_hist_ques', None)
        self.max_len_ans = truncate_config.get('max_len_ans', None)

        assert mode in ("bi")
        self.mode = mode
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix
        self.visdial_v = visdial_v
        self.pad_hist = pad_hist
        self.inc_full_hist = inc_full_hist
        if inc_full_hist:
            print('Include all the history')

        if visdial_v == '0.9' and self.len_vis_input == 100:
            self.id2img = pickle.load(open('caption_dist/id2img.pkl', 'rb'))
        elif self.len_vis_input == 36:
            self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory=True)
        else:
            raise NotImplementedError

        if mode == 's2s':
            self.task_idx = 3  # relax projection layer for different tasks [yue: just reserve this, no effects]
        elif mode == 'bi':
            self.task_idx = 0
        else:
            raise NotImplementedError

    def __call__(self, instance):
        # (img_id, cap_tokens, ques_tokens, ans_tokens, ans_opts_tokens)
        img_path, cap_tokens, ques_tokens_turns, ans_tokens_turns, ans_opts_tokens = instance
        tokens_a = ['[UNK]'] * self.len_vis_input

        def pad_to_length(tokens, length):
            tokens = tokens[:length]
            if len(tokens) < length:
                tokens += ['[PAD]'] * (length - len(tokens))
            return tokens

        turn_num = len(ques_tokens_turns)

        # if turn_i == 0:
        #     hist_tokens = cap_tokens[:20]
        # else:
        #     prev_ans = ans_tokens_turns[turn_i - 1][:8]
        #     prev_ques = ques_tokens_turns[turn_i - 1][:20 - len(prev_ans)]
        #     if self.inc_full_hist:
        #         hist_tokens += ['[SEP_0]'] + prev_ques + prev_ans
        #     else:
        #         hist_tokens = prev_ques + prev_ans
        # cur_hist = copy.deepcopy(hist_tokens)

        hist_tokens = cap_tokens[:20]

        if turn_num > 1:
            if self.inc_full_hist:
                for turn_i in range(turn_num - 1):
                    prev_ans = ans_tokens_turns[turn_i][:8]
                    prev_ques = ques_tokens_turns[turn_i][:20 - len(prev_ans)]
                    hist_tokens += ['[SEP_0]'] + prev_ques + prev_ans
            else:
                prev_ans = ans_tokens_turns[turn_num - 2][:8]
                prev_ques = ques_tokens_turns[turn_num - 2][:20 - len(prev_ans)]
                hist_tokens = prev_ques + prev_ans

        ques_tokens = ques_tokens_turns[turn_num - 1]

        if len(ques_tokens) < self.max_len_hist_ques:
            if self.pad_hist:
                hist_tokens = pad_to_length(hist_tokens, self.max_len_hist_ques - len(ques_tokens))
            else:
                hist_tokens = hist_tokens[:self.max_len_hist_ques - len(ques_tokens)]
        else:
            hist_tokens = []
            ques_tokens = ques_tokens[:self.max_len_hist_ques]

        prev_tokens = hist_tokens + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
        if self.pad_hist:
            assert len(prev_tokens) == self.max_len_hist_ques + 2
        else:
            assert len(prev_tokens) <= self.max_len_hist_ques + 2

        input_ids_opt = []
        segment_ids_opt = []
        input_mask_opt = []

        for opt_id, cur_ans_opt in enumerate(ans_opts_tokens):
            cur_ans_opt = cur_ans_opt[:self.max_len_ans]
            ans_tokens = cur_ans_opt[:self.max_len_ans]

            tokens_b = prev_tokens + ans_tokens
            assert len(tokens_b) <= self.max_len_hist_ques + 2 + self.max_len_ans

            # Add Special Tokens
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            assert len(tokens) <= self.max_len

            if self.new_segment_ids:
                if self.mode == 's2s':
                    # segment_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (
                            len(tokens_b) + 1)  # yue, do not new_segment_ids
                elif self.mode == 'bi':
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                # elif self.mode == 'l2r':
                #     segment_ids = [2] * (len(tokens))
            else:
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            input_ids = self.indexer(tokens)

            input_mask = self.get_attn_mask(tokens, len(prev_tokens))
            n_pad = self.max_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            segment_ids.extend([0] * n_pad)

            input_ids_opt.append(input_ids)
            segment_ids_opt.append(segment_ids)
            input_mask_opt.append(input_mask)

        input_mask_opt = torch.stack(input_mask_opt)
        img, vis_pe = self.get_butd(img_path)
        return (input_ids_opt, segment_ids_opt, input_mask_opt, self.task_idx, img, vis_pe)

    def get_attn_mask(self, tokens, prev_tokens_len):
        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

        if self.mode == "bi":
            input_mask[:, :len(tokens)].fill_(1)
        elif self.mode == "s2s":
            if self.only_mask_ans:
                input_mask[:, :self.len_vis_input + 2 + prev_tokens_len].fill_(1)
                second_st, second_end = self.len_vis_input + 2 + prev_tokens_len, len(tokens)
            else:
                input_mask[:, :self.len_vis_input + 2].fill_(1)
                second_st, second_end = self.len_vis_input + 2, len(tokens)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])

        if self.pad_hist:
            padded_pos = [id for id, t in enumerate(tokens) if t == '[PAD]']
            if len(padded_pos) > 0:
                assert len(padded_pos) == padded_pos[-1] - padded_pos[0] + 1
                input_mask[:, padded_pos[0]:padded_pos[-1] + 1].fill_(0)
        #
        # if self.mask_image_regions:
        #     vis_masked_pos = np.random.choice(self.len_vis_input,
        #                                       int(self.len_vis_input * self.vis_mask_prob),
        #                                       replace=False) + 1  # +1 for [CLS], always of the same length
        # else:
        #     vis_masked_pos = []
        #
        # if self.mask_image_regions:
        #     input_mask[:, vis_masked_pos].fill_(0)  # block the masked visual feature
        return input_mask

    def get_butd(self, img_id):
        if self.len_vis_input == 100 and self.visdial_v == '0.9':
            last_3_id_str = '{:0>12d}'.format(img_id)[-3:]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix + '_feat' + last_3_id_str + '.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix + '_cls' + last_3_id_str + '.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img_id_key = self.id2img[img_id]  # example of img_id_key: COCO_train2014_00000000000
                    img = torch.from_numpy(region_feat_f[img_id_key][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id_key][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id_key][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1)  # confident score
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                                F.layer_norm(cls_label, [1601])), dim=-1)  # 1601 hard coded...
        elif self.len_vis_input == 36:
            img, boxes, single_class, single_score = self.hdf_reader[img_id]
            img = torch.Tensor(img)
            boxes = torch.Tensor(boxes)
            single_class = torch.Tensor(single_class)
            single_score = torch.Tensor(single_score)

            w_est = torch.max(boxes[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(boxes[:, [1, 3]]) * 1. + 1e-5
            boxes[:, [0, 2]] /= w_est
            boxes[:, [1, 3]] /= h_est
            rel_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((boxes[:, :4], rel_area.view(-1, 1), single_class.view(-1, 1), single_score.view(-1, 1)),
                               -1)  # 7 dim
        else:
            raise NotImplementedError
        return img, vis_pe


class Preprocess4TestVisdialGen(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, new_segment_ids=False,
                 truncate_config={}, mode="s2s",
                 region_bbox_file='', region_det_file_prefix='', image_features_hdfpath='', visdial_v='1.0',
                 pad_hist=False, inc_full_hist=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids

        self.len_vis_input = truncate_config.get('len_vis_input', None)
        self.max_len_hist_ques = truncate_config.get('max_len_hist_ques', None)
        self.max_len_ans = truncate_config.get('max_len_ans', None)

        assert mode in ("s2s")
        self.mode = mode
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix
        self.visdial_v = visdial_v
        self.pad_hist = pad_hist
        self.inc_full_hist = inc_full_hist

        if visdial_v == '0.9' and self.len_vis_input == 100:
            self.id2img = pickle.load(open('caption_dist/id2img.pkl', 'rb'))
        elif self.len_vis_input == 36:
            self.hdf_reader = ImageFeaturesHdfReader(image_features_hdfpath, in_memory=True)
        else:
            raise NotImplementedError

        if mode == 's2s':
            self.task_idx = 3  # relax projection layer for different tasks [yue: just reserve this, no effects]
        elif mode == 'bi':
            self.task_idx = 0
        else:
            raise NotImplementedError

    def __call__(self, instance):
        img_path, cap_tokens, ques_tokens_turns, ans_tokens_turns, ans_opts_tokens_turns = instance
        tokens_a = ['[UNK]'] * self.len_vis_input

        def pad_to_length(tokens, length):
            tokens = tokens[:length]
            if len(tokens) < length:
                tokens += ['[PAD]'] * (length - len(tokens))
            return tokens

        input_ids_turn = []
        segment_ids_turn = []
        input_mask_turn = []
        position_ids_turn = []
        ans_ids_turn = []
        ans_opts_ids_turn = []
        turn_num = len(ques_tokens_turns)
        for turn_i in range(turn_num):
            if turn_i == 0:
                hist_tokens = cap_tokens[:20]
            else:
                prev_ans = ans_tokens_turns[turn_i - 1][:8]
                prev_ques = ques_tokens_turns[turn_i - 1][:20 - len(prev_ans)]
                if self.inc_full_hist:
                    hist_tokens += ['[SEP_0]'] + prev_ques + prev_ans
                else:
                    hist_tokens = prev_ques + prev_ans
            cur_hist = copy.deepcopy(hist_tokens)

            ques_tokens = ques_tokens_turns[turn_i]

            if len(ques_tokens) < self.max_len_hist_ques:
                if self.pad_hist:
                    cur_hist = pad_to_length(cur_hist, self.max_len_hist_ques - len(ques_tokens))
                else:
                    cur_hist = cur_hist[:self.max_len_hist_ques - len(ques_tokens)]
            else:
                cur_hist = []
                ques_tokens = ques_tokens[:self.max_len_hist_ques]

            prev_tokens = cur_hist + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']

            # if turn_i == 0:
            #     hist_tokens = cap_tokens
            # else:
            #     hist_tokens = ques_tokens_turns[turn_i - 1] + ans_tokens_turns[turn_i - 1]
            #
            # ques_tokens = ques_tokens_turns[turn_i]
            #
            # if len(ques_tokens) < self.max_len_hist_ques:
            #     if self.pad_hist:
            #         hist_tokens = pad_to_length(hist_tokens, self.max_len_hist_ques - len(ques_tokens))
            #     else:
            #         hist_tokens = hist_tokens[:self.max_len_hist_ques - len(ques_tokens)]
            # else:
            #     hist_tokens = []
            #     ques_tokens = ques_tokens[:self.max_len_hist_ques]
            #
            # prev_tokens = hist_tokens + ['[SEP_0]'] + ques_tokens + ['[SEP_1]']
            # currently do not support pad history in generative decoding setting (or need to change the input mask)
            assert not self.pad_hist and len(prev_tokens) <= self.max_len_hist_ques + 2

            tokens_b = pad_to_length(prev_tokens, self.max_len_hist_ques + 2)

            # Add Special Tokens
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b
            assert len(tokens) == self.max_len - self.max_len_ans

            assert self.mode == 's2s'
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (self.max_len_hist_ques + 2 + self.max_len_ans)
            input_ids = self.indexer(tokens)

            input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
            input_mask[:, :self.len_vis_input + 2 + self.max_len_hist_ques + 2].fill_(1)
            second_st, second_end = self.len_vis_input + 2 + self.max_len_hist_ques + 2, self.max_len
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])

            padded_pos = [id for id, t in enumerate(tokens) if t == '[PAD]']
            if len(padded_pos) > 0:
                assert len(padded_pos) == padded_pos[-1] - padded_pos[0] + 1
                input_mask[:, padded_pos[0]:padded_pos[-1] + 1].fill_(0)

            # Need to align them for decoding, use position ids to identify
            position_ids = []
            for idx in range(self.max_len):
                if idx < self.len_vis_input + 2 + len(prev_tokens):
                    position_ids.append(idx)
                elif idx >= self.len_vis_input + 2 + self.max_len_hist_ques + 2:
                    shift_idx = idx - (self.max_len_hist_ques + 2 - len(prev_tokens))
                    position_ids.append(shift_idx)
                else:
                    position_ids.append(0)

            ans_ids = self.indexer(pad_to_length(ans_tokens_turns[turn_i], self.max_len_ans))
            ans_opts_ids = [self.indexer(pad_to_length(ans_opt, self.max_len_ans))
                            for ans_opt in ans_opts_tokens_turns[turn_i]]
            # only input_ids are shorter than others
            input_ids_turn.append(input_ids)
            segment_ids_turn.append(segment_ids)
            input_mask_turn.append(input_mask)
            position_ids_turn.append(position_ids)
            ans_ids_turn.append(ans_ids)
            ans_opts_ids_turn.append(ans_opts_ids)

        img, vis_pe = self.get_butd(img_path)
        input_mask_turn = torch.stack(input_mask_turn)
        return (input_ids_turn, segment_ids_turn, position_ids_turn, ans_ids_turn, ans_opts_ids_turn,
                input_mask_turn, self.task_idx, img, vis_pe)

    def get_butd(self, img_id):
        if self.len_vis_input == 100 and self.visdial_v == '0.9':
            last_3_id_str = '{:0>12d}'.format(img_id)[-3:]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix + '_feat' + last_3_id_str + '.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix + '_cls' + last_3_id_str + '.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img_id_key = self.id2img[img_id]  # example of img_id_key: COCO_train2014_00000000000
                    img = torch.from_numpy(region_feat_f[img_id_key][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id_key][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id_key][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(vis_pe[:, [1, 3]]) * 1. + 1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3] - vis_pe[:, 1]) * (vis_pe[:, 2] - vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1)  # confident score
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                                F.layer_norm(cls_label, [1601])), dim=-1)  # 1601 hard coded...
        elif self.len_vis_input == 36:
            img, boxes, single_class, single_score = self.hdf_reader[img_id]
            img = torch.Tensor(img)
            boxes = torch.Tensor(boxes)
            single_class = torch.Tensor(single_class)
            single_score = torch.Tensor(single_score)

            w_est = torch.max(boxes[:, [0, 2]]) * 1. + 1e-5
            h_est = torch.max(boxes[:, [1, 3]]) * 1. + 1e-5
            boxes[:, [0, 2]] /= w_est
            boxes[:, [1, 3]] /= h_est
            rel_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((boxes[:, :4], rel_area.view(-1, 1), single_class.view(-1, 1), single_score.view(-1, 1)),
                               -1)  # 7 dim
        else:
            raise NotImplementedError
        return img, vis_pe


