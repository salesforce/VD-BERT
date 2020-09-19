"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import json
import argparse
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import copy
import time

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vdbert.loader_utils import batch_list_to_batch_tensors
from vdbert.seq2seq_loader import Preprocess4TrainVisdial, VisdialDataset
from vdbert.data_parallel import DataParallelImbalance


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_file", type=str, help="The input data file name.")
    parser.add_argument("--neg_num", default=0, type=int)
    parser.add_argument("--only_qa", default=0, type=int)
    parser.add_argument("--no_h0", default=0, type=int)
    parser.add_argument("--no_vision", default=0, type=int)
    parser.add_argument("--sub_sample", default=0, type=int)
    parser.add_argument("--adaptive_weight", default=0, type=int)
    parser.add_argument("--multiple_neg", default=0, type=int)
    parser.add_argument("--inc_gt_rel", default=0, type=int)
    parser.add_argument("--inc_full_hist", default=0, type=int)
    parser.add_argument("--just_for_pretrain", default=0, type=int)
    parser.add_argument("--add_boundary", default=0, type=int)
    parser.add_argument("--add_attn_fuse", default=0, type=int)
    parser.add_argument("--pad_hist", default=0, type=int)
    parser.add_argument("--only_mask_ans", default=0, type=int)
    parser.add_argument("--visdial_v", default='1.0', choices=['1.0', '0.9'], type=str)
    parser.add_argument("--loss_type", default='mlm', choices=['mlm', 'nsp', 'mlm_nsp'], type=str)
    parser.add_argument("--image_features_hdfpath", default='/export/home/vlp_data/visdial/img_feats1.0/train.h5',
                        type=str)

    parser.add_argument('--len_vis_input', type=int, default=36)
    parser.add_argument('--max_len_ans', type=int, default=10)
    parser.add_argument('--max_len_hist_ques', type=int, default=40)

    parser.add_argument("--finetune", default=0, type=int)
    parser.add_argument('--tasks', default='img2txt',
                        help='img2txt | vqa2| ctrl2 | visdial | visdial_short_hist | visdial_nsp')

    # General
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--output_dir",
                        default='tmp',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--global_rank",
                        type=int,
                        default=-1,
                        help="global_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=3,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=4, type=int,  # yue should be 4
                        help="Number of workers for the data loader.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")

    # Others for VLP
    parser.add_argument('--enable_visdom', action='store_true')
    parser.add_argument('--visdom_port', type=int, default=8888)
    # parser.add_argument('--resnet_model', type=str, default='imagenet_weights/resnet101.pth')
    parser.add_argument('--image_root', type=str, default='/mnt/dat/COCO/images')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='coco | flickr30k | cc')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'restval'])

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='file://[PT_OUTPUT_DIR]/nonexistent_file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--file_valid_jpgs', default='/mnt/dat/COCO/annotations/coco_valid_jpgs.json', type=str)
    parser.add_argument('--sche_mode', default='warmup_linear', type=str,
                        help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--use_num_imgs', default=500, type=int)
    parser.add_argument('--vis_mask_prob', default=0, type=float)
    parser.add_argument('--max_drop_worst_ratio', default=0, type=float)
    parser.add_argument('--drop_after', default=6, type=int)

    parser.add_argument('--s2s_prob', default=1, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq).")
    parser.add_argument('--bi_prob', default=0, type=float,
                        help="Percentage of examples that are bidirectional LM.")
    parser.add_argument('--l2r_prob', default=0, type=float,
                        help="Percentage of examples that are unidirectional (left-to-right) LM.")
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5',
                        type=str)
    parser.add_argument('--region_det_file_prefix',
                        default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--scst', action='store_true',
                        help='Self-critical sequence training')

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(d) for d in range(args.world_size)])

    print('Arguments: %s' % (' '.join(sys.argv[:])))
    return args


def main():
    args = process_args()
    if args.loss_type == 'mlm':
        assert args.neg_num == 0 and args.multiple_neg == 0
    elif args.loss_type == 'nsp':
        assert int(args.bi_prob) == 1 and args.max_pred == 0 and args.neg_num > 0

    if args.adaptive_weight == 1:
        assert args.neg_num > 1
    if args.add_boundary == 1:
        assert args.inc_full_hist

    if args.world_size > 1:
        print('global_rank: {}, local rank: {}'.format(args.global_rank, args.local_rank))

    # Input format: [CLS] img [SEP] hist [SEP_0] ques [SEP_1] ans [SEP]
    args.max_seq_length = args.len_vis_input + 2 + args.max_len_hist_ques + 2 + args.max_len_ans + 1

    args.mask_image_regions = (args.vis_mask_prob > 0)  # whether to mask out image regions
    args.dist_url = args.dist_url.replace('[PT_OUTPUT_DIR]', args.output_dir)

    # arguments inspection
    assert args.enable_butd, 'only support region attn! featmap attn deprecated'

    if args.enable_butd:
        if args.visdial_v == '1.0':
            assert (args.len_vis_input == 36) or (args.len_vis_input == 0)
        elif args.visdial_v == '0.9':
            if (args.len_vis_input == 100):
                args.region_bbox_file = os.path.join(args.image_root, args.region_bbox_file)
                args.region_det_file_prefix = os.path.join(args.image_root,
                                                           args.region_det_file_prefix) if args.dataset in (
                    'cc', 'coco') and args.region_det_file_prefix != '' else ''

    # output config
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_file),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s'))
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.global_rank)
    logger.info('Arguments: %s\n' % (' '.join(sys.argv[:])))
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # plotting loss, optional
    if args.enable_visdom:
        import visdom
        vis = visdom.Visdom(port=args.visdom_port, env=args.output_dir)
        vis_window = {'iter': None, 'score': None}

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case,
        cache_dir=args.output_dir + '/.pretrained_model_{}'.format(args.global_rank))
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
    assert args.do_train
    logger.info('Max seq length: %d, batch size: %d\n' % (args.max_seq_length, args.train_batch_size))
    bi_uni_pipeline = [Preprocess4TrainVisdial(args.max_pred, args.mask_prob,
                                               list(tokenizer.vocab.keys()),
                                               tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                               new_segment_ids=args.new_segment_ids,
                                               truncate_config={'len_vis_input': args.len_vis_input,
                                                                'max_len_hist_ques': args.max_len_hist_ques,
                                                                'max_len_ans': args.max_len_ans},
                                               mask_image_regions=args.mask_image_regions,
                                               mode="s2s", vis_mask_prob=args.vis_mask_prob,
                                               region_bbox_file=args.region_bbox_file,
                                               region_det_file_prefix=args.region_det_file_prefix,
                                               image_features_hdfpath=args.image_features_hdfpath,
                                               visdial_v=args.visdial_v, pad_hist=args.pad_hist,
                                               finetune=args.finetune,
                                               only_mask_ans=args.only_mask_ans,
                                               add_boundary=args.add_boundary,
                                               only_qa=args.only_qa),
                       Preprocess4TrainVisdial(args.max_pred, args.mask_prob,
                                               list(tokenizer.vocab.keys()),
                                               tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                               new_segment_ids=args.new_segment_ids,
                                               truncate_config={'len_vis_input': args.len_vis_input,
                                                                'max_len_hist_ques': args.max_len_hist_ques,
                                                                'max_len_ans': args.max_len_ans},
                                               mask_image_regions=args.mask_image_regions,
                                               mode="bi", vis_mask_prob=args.vis_mask_prob,
                                               region_bbox_file=args.region_bbox_file,
                                               region_det_file_prefix=args.region_det_file_prefix,
                                               image_features_hdfpath=args.image_features_hdfpath,
                                               visdial_v=args.visdial_v, pad_hist=args.pad_hist,
                                               finetune=args.finetune,
                                               only_mask_ans=args.only_mask_ans,
                                               add_boundary=args.add_boundary,
                                               only_qa=args.only_qa)]

    train_dataset = VisdialDataset(
        args.src_file, args.train_batch_size, data_tokenizer, use_num_imgs=args.use_num_imgs,
        bi_uni_pipeline=bi_uni_pipeline, s2s_prob=args.s2s_prob, bi_prob=args.bi_prob,
        is_train=args.do_train, neg_num=args.neg_num, inc_gt_rel=args.inc_gt_rel,
        inc_full_hist=args.inc_full_hist, just_for_pretrain=args.just_for_pretrain, sub_sample=args.sub_sample)

    if args.world_size == 1:
        train_sampler = RandomSampler(train_dataset, replacement=False)
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size, sampler=train_sampler,
                                                   num_workers=args.num_workers,
                                                   collate_fn=batch_list_to_batch_tensors, pin_memory=True)

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    t_total = int(len(train_dataloader) * args.num_train_epochs * 1. /
                  args.gradient_accumulation_steps)

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3 if args.tasks == 'img2txt' else 0
    mask_word_id, eos_word_ids, pad_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[PAD]"])  # index in BERT vocab: 103, 102, 0

    if (args.model_recover_path is None):
        # if _state_dict == {}, the parameters are randomly initialized
        # if _state_dict == None, the parameters are initialized with bert-init
        assert args.scst == False, 'must init from maximum likelihood training'
        _state_dict = {} if args.from_scratch else None
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=_state_dict, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding,
            cache_dir=args.output_dir + '/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob, enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, visdial_v=args.visdial_v, loss_type=args.loss_type,
            neg_num=args.neg_num, adaptive_weight=args.adaptive_weight, add_attn_fuse=args.add_attn_fuse,
            no_h0=args.no_h0, no_vision=args.no_vision)
        global_step = 0
    else:
        if args.model_recover_path:
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)
            model_recover = torch.load(args.model_recover_path)
            global_step = 0

        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding,
            cache_dir=args.output_dir + '/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob, enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, visdial_v=args.visdial_v, loss_type=args.loss_type,
            neg_num=args.neg_num, adaptive_weight=args.adaptive_weight, add_attn_fuse=args.add_attn_fuse,
            no_h0=args.no_h0, no_vision=args.no_vision)

        del model_recover
        torch.cuda.empty_cache()

    if args.fp16:
        model.half()
        # cnn.half()
        if args.fp32_embedding:
            model.bert.embeddings.word_embeddings.float()
            model.bert.embeddings.position_embeddings.float()
            model.bert.embeddings.token_type_embeddings.float()
    model.to(device)
    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        model = DataParallelImbalance(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer_State(
                optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer_State(
                optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             schedule=args.sche_mode,
                             t_total=t_total)


    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
        logger.info("  Loader length = %d", len(train_dataloader))

        model.train()

        start_epoch = 1
        logger.info("Begin training from epoch = %d", start_epoch)
        t0 = time.time()
        for i_epoch in trange(start_epoch, args.num_train_epochs + 1, desc="Epoch"):
            if args.multiple_neg and i_epoch > 1:
                train_dataset = VisdialDataset(
                    args.src_file, args.train_batch_size, data_tokenizer, use_num_imgs=args.use_num_imgs,
                    bi_uni_pipeline=bi_uni_pipeline, s2s_prob=args.s2s_prob, bi_prob=args.bi_prob,
                    is_train=args.do_train, neg_num=args.neg_num, inc_gt_rel=args.inc_gt_rel,
                    inc_full_hist=args.inc_full_hist, just_for_pretrain=args.just_for_pretrain, sub_sample=args.sub_sample)

                if args.world_size == 1:
                    train_sampler = RandomSampler(train_dataset, replacement=False)
                else:
                    train_sampler = DistributedSampler(train_dataset)

                train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                               batch_size=args.train_batch_size, sampler=train_sampler,
                                                               num_workers=args.num_workers,
                                                               collate_fn=batch_list_to_batch_tensors, pin_memory=True)
            if args.local_rank >= 0:
                train_sampler.set_epoch(i_epoch - 1)
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
            nbatches = len(train_dataloader)
            losses = []
            pretext_loss = []
            mlm_losses = []
            nsp_losses = []
            for step, batch in enumerate(iter_bar):
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, is_next, \
                task_idx, vis_masked_pos, img, vis_pe = batch

                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()

                if args.enable_butd:
                    conv_feats = img.data  # Bx100x2048
                    vis_pe = vis_pe.data

                loss_tuple = model(conv_feats, vis_pe, input_ids, segment_ids,
                                   input_mask, lm_label_ids, is_next, masked_pos=masked_pos,
                                   masked_weights=masked_weights, task_idx=task_idx,
                                   vis_masked_pos=vis_masked_pos, mask_image_regions=args.mask_image_regions,
                                   drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0)

                # disable pretext_loss_deprecated for now
                masked_lm_loss, pretext_loss_deprecated, nsp_loss = loss_tuple
                if n_gpu > 1:  # mean() to average on multi-gpu. For dist, this is done through gradient addition.
                    masked_lm_loss = masked_lm_loss.mean()
                    pretext_loss_deprecated = pretext_loss_deprecated.mean()
                    nsp_loss = nsp_loss.mean()
                loss = masked_lm_loss + pretext_loss_deprecated + nsp_loss

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
                losses.append(loss.item())
                mlm_losses.append(masked_lm_loss.item())
                pretext_loss.append(pretext_loss_deprecated.item())
                nsp_losses.append(nsp_loss.item())

                if step % max(1, nbatches // 10) == 0:
                    logger.info(
                        "Epoch {}, Iter {}, Loss {:.4f}, MLM {:.4f}, NSP {:.4f}, Elapse time {:.2f}\n".format(
                            i_epoch, step, np.mean(losses), np.mean(mlm_losses), np.mean(nsp_losses), time.time() - t0))

                if args.enable_visdom:
                    if vis_window['iter'] is None:
                        vis_window['iter'] = vis.line(
                            X=np.tile(np.arange((i_epoch - 1) * nbatches + step,
                                                (i_epoch - 1) * nbatches + step + 1), (1, 1)).T,
                            Y=np.column_stack((np.asarray([np.mean(losses)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total'])
                        )
                    else:
                        vis.line(
                            X=np.tile(np.arange((i_epoch - 1) * nbatches + step,
                                                (i_epoch - 1) * nbatches + step + 1), (1, 1)).T,
                            Y=np.column_stack((np.asarray([np.mean(losses)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total']),
                            win=vis_window['iter'],
                            update='append'
                        )

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                    if amp_handle:
                        amp_handle._clear_cache()
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_this_step = args.learning_rate * \
                                   warmup_linear(global_step / t_total,
                                                 args.warmup_proportion)
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Save a trained model
            logger.info(
                "** ** * Saving fine-tuned model and optimizer ** ** * ")
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                args.output_dir, "model.%d.%.3f.bin" % (i_epoch, np.mean(losses)))
            if args.global_rank in (-1, 0):  # save model if the first device or no dist
                torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)
                logger.info("Save model to %s", output_model_file)
            logger.info("Finish training epoch %d, avg loss: %.2f and takes %.2f seconds" % (
                i_epoch, np.mean(losses), time.time() - t0))
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            if args.world_size > 1:
                torch.distributed.barrier()


if __name__ == "__main__":
    main()
