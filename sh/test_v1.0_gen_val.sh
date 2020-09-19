checkpoint_fn=saved_models/v1.0_mlm_gen

epoch=11.0.961
split=val
len_vis_input=36
visdial_v=1.0
use_num_imgs=2
batch_size=4

if [[ $use_num_imgs == -1 ]]; then
  img_num_tag=full
else
  img_num_tag=$use_num_imgs
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

WORK_DIR=/export/share/yuewang/VD-BERT-Clean
export CHECKPOINT_ROOT=${WORK_DIR}/checkpoints
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
model_path=${CHECKPOINT_ROOT}/${checkpoint_fn}/model.${epoch}.bin


python vdbert/test_visdial_gen_val.py \
  --model_recover_path  ${model_path} \
  --new_segment_ids --beam_size 1 --enable_butd --visdial_v ${visdial_v} --split ${split} \
  --src_file  ${WORK_DIR}/data/visdial_${visdial_v}_${split}.json \
  --image_features_hdfpath  ${WORK_DIR}/data/img_feats1.0/${split}.h5 \
  --save_ranks_path ${CHECKPOINT_ROOT}/${checkpoint_fn}/${split}_${img_num_tag}_ranks_model${epoch}.json \
  --use_num_imgs ${use_num_imgs} --batch_size ${batch_size} --inc_full_hist 1 \
  --len_vis_input ${len_vis_input} --max_len_hist_ques 200 --max_len_ans 10 --decode_verbose 0 \
  --gt_rel_file ${WORK_DIR}/data/visdial_${visdial_v}_${split}_dense_annotations.json

