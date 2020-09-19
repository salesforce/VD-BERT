loss_type=nsp
split=train_val_rel
len_vis_input=36
seed=42
use_num_imgs=10
bs=1
rank_loss=softmax  # ['softmax', 'listmle', 'listnet', 'approxndcg']

checkpoint_output=v1.0_${loss_type}_disc_dense_${rank_loss}_s${seed}_g4

#export CUDA_VISIBLE_DEVICES=0,1,2,3

WORK_DIR=/export/share/yuewang/VD-BERT-Clean
export CHECKPOINT_ROOT=${WORK_DIR}/checkpoints
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
model_path=${CHECKPOINT_ROOT}/saved_models/v1.0_nsp_disc/model.9.0.122.bin


python vdbert/train_visdial_disc_dense.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --model_recover_path ${model_path} --len_vis_input ${len_vis_input}  \
    --do_train --new_segment_ids --enable_butd --visdial_v 1.0 \
    --train_src_file ${WORK_DIR}/data/visdial_1.0_train.json \
    --val_src_file ${WORK_DIR}/data/visdial_1.0_val.json \
    --train_rel_file ${WORK_DIR}/data/visdial_1.0_train_dense_sample.json \
    --val_rel_file ${WORK_DIR}/data/visdial_1.0_val_dense_annotations.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 0   \
    --max_len_hist_ques 200 --max_len_ans 10 --multiple_neg 1 --neg_num 1  --inc_full_hist 1 \
    --float_nsp_label 1 --inc_gt_rel 1 --rank_loss ${rank_loss} --seed ${seed} \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 10 \
    --local_rank 0 --global_rank 0 --world_size 4 & \
python vdbert/train_visdial_disc_dense.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --model_recover_path ${model_path} --len_vis_input ${len_vis_input}  \
    --do_train --new_segment_ids --enable_butd --visdial_v 1.0 \
    --train_src_file ${WORK_DIR}/data/visdial_1.0_train.json \
    --val_src_file ${WORK_DIR}/data/visdial_1.0_val.json \
    --train_rel_file ${WORK_DIR}/data/visdial_1.0_train_dense_sample.json \
    --val_rel_file ${WORK_DIR}/data/visdial_1.0_val_dense_annotations.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 0   \
    --max_len_hist_ques 200 --max_len_ans 10 --multiple_neg 1 --neg_num 1  --inc_full_hist 1 \
    --float_nsp_label 1 --inc_gt_rel 1 --rank_loss ${rank_loss} --seed ${seed} \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 10 \
    --local_rank 1 --global_rank 1 --world_size 4 & \
python vdbert/train_visdial_disc_dense.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --model_recover_path ${model_path} --len_vis_input ${len_vis_input}  \
    --do_train --new_segment_ids --enable_butd --visdial_v 1.0 \
    --train_src_file ${WORK_DIR}/data/visdial_1.0_train.json \
    --val_src_file ${WORK_DIR}/data/visdial_1.0_val.json \
    --train_rel_file ${WORK_DIR}/data/visdial_1.0_train_dense_sample.json \
    --val_rel_file ${WORK_DIR}/data/visdial_1.0_val_dense_annotations.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 0   \
    --max_len_hist_ques 200 --max_len_ans 10 --multiple_neg 1 --neg_num 1  --inc_full_hist 1 \
    --float_nsp_label 1 --inc_gt_rel 1 --rank_loss ${rank_loss} --seed ${seed} \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 10 \
    --local_rank 2 --global_rank 2 --world_size 4 & \
python vdbert/train_visdial_disc_dense.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --model_recover_path ${model_path} --len_vis_input ${len_vis_input}  \
    --do_train --new_segment_ids --enable_butd --visdial_v 1.0 \
    --train_src_file ${WORK_DIR}/data/visdial_1.0_train.json \
    --val_src_file ${WORK_DIR}/data/visdial_1.0_val.json \
    --train_rel_file ${WORK_DIR}/data/visdial_1.0_train_dense_sample.json \
    --val_rel_file ${WORK_DIR}/data/visdial_1.0_val_dense_annotations.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 0   \
    --max_len_hist_ques 200 --max_len_ans 10 --multiple_neg 1 --neg_num 1  --inc_full_hist 1 \
    --float_nsp_label 1 --inc_gt_rel 1 --rank_loss ${rank_loss} --seed ${seed} \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 10 \
    --local_rank 3 --global_rank 3 --world_size 4
