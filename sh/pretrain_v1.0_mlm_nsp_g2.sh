visdial_v=1.0
loss_type=mlm_nsp
split=train
bs=30
use_num_imgs=10

checkpoint_output=v${visdial_v}_${loss_type}_from_BERT_g4

export CUDA_VISIBLE_DEVICES=0,1,2,3

WORK_DIR=/export/share/yuewang/VD-BERT-Clean
export CHECKPOINT_ROOT=${WORK_DIR}/checkpoints
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
model_path=${CHECKPOINT_ROOT}/saved_models/v1.0_from_BERT_e30.bin

python vdbert/train_visdial.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --do_train --new_segment_ids --enable_butd --visdial_v ${visdial_v} \
    --src_file ${WORK_DIR}/data/visdial_${visdial_v}_${split}.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 30 --neg_num 1 --multiple_neg 1 \
    --inc_full_hist 1  --max_len_hist_ques 200 --max_len_ans 10 \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 20 \
    --local_rank 0 --global_rank 0 --world_size 4 & \
python vdbert/train_visdial.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --do_train --new_segment_ids --enable_butd --visdial_v ${visdial_v} \
    --src_file ${WORK_DIR}/data/visdial_${visdial_v}_${split}.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 30 --neg_num 1 --multiple_neg 1 \
    --inc_full_hist 1  --max_len_hist_ques 200 --max_len_ans 10 \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 20 \
    --local_rank 1 --global_rank 1 --world_size 4 & \
python vdbert/train_visdial.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --do_train --new_segment_ids --enable_butd --visdial_v ${visdial_v} \
    --src_file ${WORK_DIR}/data/visdial_${visdial_v}_${split}.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 30 --neg_num 1 --multiple_neg 1 \
    --inc_full_hist 1  --max_len_hist_ques 200 --max_len_ans 10 \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 20 \
    --local_rank 2 --global_rank 2 --world_size 4 & \
python vdbert/train_visdial.py \
    --output_dir ${CHECKPOINT_ROOT}/${checkpoint_output} \
    --do_train --new_segment_ids --enable_butd --visdial_v ${visdial_v} \
    --src_file ${WORK_DIR}/data/visdial_${visdial_v}_${split}.json \
    --image_features_hdfpath ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
    --s2s_prob 0 --bi_prob 1 --loss_type ${loss_type} --max_pred 30 --neg_num 1 --multiple_neg 1 \
    --inc_full_hist 1  --max_len_hist_ques 200 --max_len_ans 10 \
    --num_workers 4 --train_batch_size ${bs}  --use_num_imgs ${use_num_imgs} --num_train_epochs 20 \
    --local_rank 3 --global_rank 3 --world_size 4 & \


