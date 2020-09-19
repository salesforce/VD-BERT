"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import json
import argparse
import math
from tqdm import tqdm
import numpy as np
import torch
import random
import sys

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForVisDialGen

from vdbert.seq2seq_loader import Preprocess4TestVisdialGen

from vdbert.metrics import scores_to_ranks, SparseGTMetrics, NDCG
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def process_args():
    print('Arguments: %s' % (' '.join(sys.argv[:])))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--decode_verbose", default=0, type=int)
    parser.add_argument("--gt_rel_file", type=str, default='')
    parser.add_argument("--src_file", type=str, help="The input data file name.")

    parser.add_argument("--neg_num", default=0, type=int)
    parser.add_argument("--inc_full_hist", default=0, type=int)
    parser.add_argument("--include_relevance", default=0, type=int)
    parser.add_argument("--pad_hist", default=0, type=int)
    parser.add_argument("--visdial_v", default='1.0', choices=['1.0', '0.9'], type=str)
    parser.add_argument("--loss_type", default='mlm', choices=['mlm', 'nsp', 'mlm_nsp'], type=str)
    parser.add_argument("--image_features_hdfpath", default='/export/home/vlp_data/visdial/img_feats1.0/train.h5',
                        type=str)

    parser.add_argument('--len_vis_input', type=int, default=36)
    parser.add_argument('--max_len_ans', type=int, default=10)
    parser.add_argument('--max_len_hist_ques', type=int, default=200)

    parser.add_argument('--save_ranks_path', default='', help='')
    parser.add_argument('--use_num_imgs', default=500, type=int)

    # General
    # parser.add_argument('--tasks', default='img2txt',
    #                     help='img2txt | vqa2| ctrl2 | visdial | visdial_short_hist | visdial_nsp')

    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")

    # For decoding
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)

    # Others for VLP
    parser.add_argument("--ref_file", default='pythia/data/v2_mscoco_val2014_annotations.json', type=str,
                        help="The annotation reference file name.")
    parser.add_argument('--dataset', default='coco', type=str,
                        help='coco | flickr30k | cc')
    parser.add_argument('--image_root', type=str, default='/mnt/dat/COCO/images')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5',
                        type=str)
    parser.add_argument('--region_det_file_prefix',
                        default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument("--output_dir",
                        default='tmp',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--file_valid_jpgs', default='', type=str)

    args = parser.parse_args()
    return args


def main():
    args = process_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.enable_butd:
        if args.visdial_v == '1.0':
            assert (args.len_vis_input == 36)
        elif args.visdial_v == '0.9':
            assert (args.len_vis_input == 100)
            args.region_bbox_file = os.path.join(args.image_root, args.region_bbox_file)
            args.region_det_file_prefix = os.path.join(args.image_root,
                                                       args.region_det_file_prefix) if args.dataset in (
                'cc', 'coco') and args.region_det_file_prefix != '' else ''

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    # for generative, there is no [SEP] at the end
    args.max_seq_length = args.len_vis_input + 2 + args.max_len_hist_ques + 2 + args.max_len_ans

    tokenizer.max_len = args.max_seq_length
    bi_uni_pipeline = [Preprocess4TestVisdialGen(list(tokenizer.vocab.keys()),
                                                 tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                                 new_segment_ids=args.new_segment_ids,
                                                 truncate_config={'len_vis_input': args.len_vis_input,
                                                                  'max_len_hist_ques': args.max_len_hist_ques,
                                                                  'max_len_ans': args.max_len_ans},
                                                 mode="s2s",
                                                 region_bbox_file=args.region_bbox_file,
                                                 region_det_file_prefix=args.region_det_file_prefix,
                                                 image_features_hdfpath=args.image_features_hdfpath,
                                                 visdial_v=args.visdial_v, pad_hist=args.pad_hist,
                                                 inc_full_hist=args.inc_full_hist)]

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]"])
    logger.info('Attempting to recover models from: {}'.format(args.model_recover_path))
    if 0 == len(glob.glob(args.model_recover_path.strip())):
        logger.error('There are no models to recover. The program will exit.')
        sys.exit(1)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)

        model = BertForVisDialGen.from_pretrained(args.bert_model,
                                                  max_position_embeddings=512,
                                                  config_path=args.config_path,
                                                  state_dict=model_recover, num_labels=cls_num_labels,
                                                  type_vocab_size=type_vocab_size, task_idx=3,
                                                  mask_word_id=mask_word_id,
                                                  search_beam_size=args.beam_size,
                                                  length_penalty=args.length_penalty,
                                                  eos_id=eos_word_ids,
                                                  ngram_size=args.ngram_size,
                                                  min_len=args.min_len,
                                                  enable_butd=args.enable_butd, len_vis_input=args.len_vis_input,
                                                  tokenizer=tokenizer, decode_verbose=args.decode_verbose,
                                                  visdial_v=args.visdial_v)
        del model_recover

        if args.fp16:
            model.half()
            # cnn.half()
        model.to(device)
        # cnn.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            # cnn = torch.nn.DataParallel(cnn)

        torch.cuda.empty_cache()
        model.eval()

        def read_data(src_file):
            eval_lst = []
            with open(src_file, "r", encoding='utf-8') as f_src:
                data = json.load(f_src)['data']
                dialogs = data['dialogs']
                questions = data['questions']
                answers = data['answers']
                img_idx = 0
                for dialog in tqdm(dialogs):
                    if img_idx < args.use_num_imgs or args.use_num_imgs == -1:
                        img_id = dialog['image_id']

                        cap_tokens = tokenizer.tokenize(dialog['caption'])

                        ques_id = [item['question'] for item in dialog['dialog']]
                        ques_tokens = [tokenizer.tokenize(questions[id] + '?') for id in ques_id]

                        ans_id = [item['answer'] for item in dialog['dialog']]
                        ans_tokens = [tokenizer.tokenize(answers[id]) for id in ans_id]
                        gt_id = [item['gt_index'] for item in dialog['dialog']]

                        ans_opts = [item['answer_options'] for item in dialog['dialog']]
                        ans_opts_tokens = [[tokenizer.tokenize(answers[id]) for id in ans] for ans in ans_opts]

                        assert len(ques_tokens) == len(ans_tokens) == len(ans_opts_tokens) == 10, \
                            "ques num: %d, ans num: %d, ans opt num: %d" % (
                                len(ques_tokens), len(ans_tokens), len(ans_opts_tokens))
                        assert all(
                            [len(ans_opt) == 100 for ans_opt in ans_opts_tokens]), "all the answer have 100 options"
                        eval_lst.append(
                            (img_id, cap_tokens, ques_tokens, ans_tokens, ans_opts_tokens, gt_id))

                        img_idx += 1
            return eval_lst

        def get_gt_rel_dict(fname):
            gt_rel_dict = {}
            gt_rel_data = json.load(open(fname))
            for item in gt_rel_data:
                image_id = item['image_id']
                round_id = item['round_id']
                gt_relevance = item['gt_relevance']
                # each image only at most has one turn having dense annotation
                if image_id not in gt_rel_dict:
                    gt_rel_dict[image_id] = (round_id, gt_relevance)
            return gt_rel_dict

        if args.gt_rel_file != '':
            gt_rel_dict = get_gt_rel_dict(args.gt_rel_file)

        input_lines = read_data(args.src_file)
        next_i = 0
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        print('start the visdial decode evaluation...')
        t0 = time.time()
        ranks_json = []
        sparse_metrics = SparseGTMetrics()
        ndcg = NDCG()
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[:-1] for x in _chunk]
                buf_gt_id = [x[-1] for x in _chunk]
                next_i += args.batch_size
                instances = []
                for instance in buf:
                    instances.append(bi_uni_pipeline[0](instance))

                with torch.no_grad():
                    buf_gt_id = torch.tensor(buf_gt_id).long().to(device)
                    batch_data = list(zip(*instances))
                    img, vis_pe = (torch.stack(x).to(device) for x in batch_data[-2:])
                    task_idx = torch.tensor(batch_data[-3], dtype=torch.long).to(device)

                    conv_feats = img.data  # Bx100x2048
                    vis_pe = vis_pe.data
                    output_scores_turn = []

                    # (input_ids_turn, segment_ids_turn, position_ids_turn, ans_ids_turn, ans_opts_ids_turn,
                    #  input_mask_turn, self.task_idx, img, vis_pe)

                    input_ids_turns = [[x[turn_i] for x in batch_data[0]] for turn_i in range(10)]
                    segment_ids_turns = [[x[turn_i] for x in batch_data[1]] for turn_i in range(10)]
                    position_ids_turns = [[x[turn_i] for x in batch_data[2]] for turn_i in range(10)]
                    ans_ids_turns = [[x[turn_i] for x in batch_data[3]] for turn_i in range(10)]
                    ans_opts_ids_turns = [[x[turn_i] for x in batch_data[4]] for turn_i in range(10)]
                    input_mask_turns = [[x[turn_i] for x in batch_data[5]] for turn_i in range(10)]

                    for turn_i in range(10):
                        input_ids = torch.tensor(input_ids_turns[turn_i], dtype=torch.long).to(device)
                        segment_ids = torch.tensor(segment_ids_turns[turn_i], dtype=torch.long).to(device)
                        position_ids = torch.tensor(position_ids_turns[turn_i], dtype=torch.long).to(device)
                        ans_ids = torch.tensor(ans_ids_turns[turn_i], dtype=torch.long).to(device)
                        ans_opts_ids = torch.tensor(ans_opts_ids_turns[turn_i], dtype=torch.long).to(device)
                        input_mask = torch.stack(input_mask_turns[turn_i]).to(device)

                        output_scores = model(conv_feats, vis_pe, input_ids, segment_ids,
                                              position_ids, input_mask, ans_ids, ans_opts_ids, task_idx=task_idx)

                        output_scores = output_scores[-1]  # [batch_size, num_options]
                        output_scores_turn.append(output_scores)

                    output_scores_turn = torch.stack(output_scores_turn, 1)  # [batch_size, num_rounds, num_options]
                    ranks = scores_to_ranks(output_scores_turn)
                    # output_scores_turn_cheat = output_scores_turn.scatter_(2, buf_gt_id.unsqueeze(2), 100.0)
                    sparse_metrics.observe(output_scores_turn, buf_gt_id)
                    for i in range(len(buf_id)):
                        # Cast into types explicitly to ensure no errors in schema.
                        # Round ids are 1-10, not 0-9
                        if args.split == "val":
                            for j in range(10):
                                ranks_json.append(
                                    {
                                        "image_id": buf_id[i],
                                        "round_id": int(j + 1),
                                        "ranks": [rank.item() for rank in ranks[i][j]],
                                    }
                                )

                    if args.gt_rel_file:
                        scores = []
                        gt_rels = []
                        for i in range(len(buf_id)):
                            if buf_id[i] in gt_rel_dict:
                                turn_idx, gt_rel = gt_rel_dict[buf_id[i]]
                                scores.append(output_scores_turn[i, turn_idx - 1, :])
                                gt_rels.append(torch.tensor(gt_rel, dtype=torch.float32).to(device))
                        scores = torch.stack(scores)
                        gt_rels = torch.stack(gt_rels)
                        ndcg.observe(scores, gt_rels)

                pbar.update(1)

        json.dump(ranks_json, open(args.save_ranks_path, "w"))
        logger.info("Finish writing rankings into %s" % (args.save_ranks_path))

        if args.split == "val":
            fw = open(args.save_ranks_path.replace('.json', '_results.txt'), "w")
            all_metrics = {}
            all_metrics.update(sparse_metrics.retrieve(reset=True))
            if args.gt_rel_file:
                all_metrics.update(ndcg.retrieve(reset=True))
            for metric_name, metric_value in all_metrics.items():
                print(f"{metric_name}: {metric_value}")
                fw.write("%s: %.6f\n" % (metric_name, metric_value))


if __name__ == "__main__":
    main()
