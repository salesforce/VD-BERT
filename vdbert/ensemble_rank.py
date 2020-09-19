import json
import os
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    res = e_x / e_x.sum(axis=0)  # only difference
    return res.tolist()


def scores_to_ranks(scores):
    """Convert model output scores into ranks."""
    assert len(scores) == 100 and isinstance(scores, list)
    ranks = [0] * 100
    ranks_id = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    for idx in range(100):
        ranks[ranks_id[idx]] = idx
    ranks = [r + 1 for r in ranks]
    return ranks


if __name__ == '__main__':
    src_fn = 'test_full_ranks_model{}_score.json'

    src_dir1 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist'
    src_dir2 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_rank_loss'
    src_dir3 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g2_disc_nsp_obj36_with_relevance_neg1_multiple'
    src_dir4 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_rank_loss_seed45'
    src_dir5 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_rank_loss_seed46'
    src_dir6 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_rank_loss_seed47'
    src_dir7 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_listnet_rank_s42'
    src_dir8 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_listnet_rank_s43'
    src_dir9 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_approxndcg_rank_s42'
    src_dir10 = '/export/home/vlp_data/new_checkpoints/visdial_1.0_g4_nsp_obj36_with_relevance_full_hist_approxndcg_rank_s45'

    epoch_tags1 = ['2.0.184', '3.0.139', '4.0.113', '5.0.095']
    epoch_tags2 = ['2.0.020', '4.0.013', '5.0.010', '6.0.009']
    epoch_tags3 = ['2.0.186', '3.0.146', '4.0.120', '5.0.102']
    epoch_tags4 = ['3.0.015', '4.0.011', '5.0.010', '6.0.008']
    epoch_tags5 = ['3.0.015', '4.0.011', '5.0.010', '6.0.009']
    epoch_tags6 = ['3.0.015', '4.0.011', '5.0.010', '6.0.008']
    epoch_tags7 = ['5.3.355', '7.3.351']
    epoch_tags8 = ['4.3.356', '5.3.355', '6.3.353']
    epoch_tags9 = ['8.-0.957']
    epoch_tags10 = ['11.-0.962']

    trg_fn = 'pred/test_full_ranks_model_assemble.json'
    normalize = True

    src_fns1 = [os.path.join(src_dir1, src_fn.format(tag)) for tag in epoch_tags1]
    src_fns2 = [os.path.join(src_dir2, src_fn.format(tag)) for tag in epoch_tags2]
    src_fns3 = [os.path.join(src_dir3, src_fn.format(tag)) for tag in epoch_tags3]
    src_fns4 = [os.path.join(src_dir4, src_fn.format(tag)) for tag in epoch_tags4]
    src_fns5 = [os.path.join(src_dir5, src_fn.format(tag)) for tag in epoch_tags5]
    src_fns6 = [os.path.join(src_dir6, src_fn.format(tag)) for tag in epoch_tags6]
    src_fns7 = [os.path.join(src_dir7, src_fn.format(tag)) for tag in epoch_tags7]
    src_fns8 = [os.path.join(src_dir8, src_fn.format(tag)) for tag in epoch_tags8]
    src_fns9 = [os.path.join(src_dir9, src_fn.format(tag)) for tag in epoch_tags9]
    src_fns10 = [os.path.join(src_dir10, src_fn.format(tag)) for tag in epoch_tags10]

    datas = [json.load(open(fn)) for fn in src_fns1 + src_fns2 + src_fns3 + src_fns4 + src_fns5 +
             src_fns6 + src_fns7 + src_fns8 + src_fns9 + src_fns10]
    trg_data = []
    for batch in zip(*datas):
        assert len(set([b['image_id'] for b in batch])) == 1
        assert len(set([b['round_id'] for b in batch])) == 1
        # scores = [0] * 100
        # for b in batch:
        #     scores += b['scores']
        if normalize:
            scores_batch = [softmax(b['scores']) for b in batch]
        else:
            scores_batch = [b['scores'] for b in batch]
        scores = [sum(item) for item in zip(*scores_batch)]
        ranks = scores_to_ranks(scores)
        example = {'image_id': batch[0]['image_id'], 'round_id': batch[0]['round_id'], 'ranks': ranks}
        trg_data.append(example)
    if normalize:
        trg_fn = trg_fn.replace('.json', '_normalize.json')
    json.dump(trg_data, open(trg_fn, "w"))
    print("Writing assemble ranks into %s" % trg_fn)

#pred/test_full_ranks_model_much_bigger_assemble_normalize.json
# {"test-std": {"MRR (x 100)": 47.33284623824266, "R@1": 33.875, "R@5": 62.2, "R@10": 78.5, "Mean": 7.00825, "NDCG (x 100)": 75.0650295646894}}

#  pred/test_full_ranks_model_much_bigger_only_best_assemble_normalize.json
#{"test-std": {"MRR (x 100)": 47.399584925133524, "R@1": 34.300000000000004, "R@5": 61.575, "R@10": 77.775, "Mean": 7.12125, "NDCG (x 100)": 74.83750529355501}}

# pred/test_full_ranks_model_new_assemble_normalize.json
# {"test-std": {"MRR (x 100)": 50.06728837554818, "R@1": 38.224999999999994, "R@5": 61.3, "R@10": 77.45, "Mean": 6.846, "NDCG (x 100)": 75.27363431534192}}

# pred/test_full_ranks_model_new_assemble1_normalize.json
# {"test-std": {"MRR (x 100)": 50.7742067478486, "R@1": 38.525, "R@5": 62.3, "R@10": 77.725, "Mean": 6.82325, "NDCG (x 100)": 75.20088773737974}}

# pred/test_full_ranks_model_new_assemble_bigger_normalize.json
# {"test-std": {"MRR (x 100)": 51.17071126367074, "R@1": 38.9, "R@5": 62.824999999999996, "R@10": 77.97500000000001, "Mean": 6.6895, "NDCG (x 100)": 75.35320108679628}}

# pred/test_full_ranks_model_new_assemble_much_bigger_normalize.json
# {"test-std": {"MRR (x 100)": 51.29660923845128, "R@1": 38.475, "R@5": 64.17500000000001, "R@10": 78.175, "Mean": 6.621, "NDCG (x 100)": 75.28807504944294}}

#pred/test_full_ranks_model_new_assemble_bigger_final_normalize.json
# {"test-std": {"MRR (x 100)": 51.30198652149388, "R@1": 38.75, "R@5": 63.725, "R@10": 77.95, "Mean": 6.662, "NDCG (x 100)": 75.33182282241258}}