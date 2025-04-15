IMAGES_ROOT="/workspace/data/images"
LABELS_ROOT="/workspace/data/labels"
PREDS_ROOT="/workspace/data/preds"
PROCESSED_PREDS_ROOT="/workspace/data/processed_preds"

MODEL="/workspace/models/yolo12x-SRSDD-competition-wtconv.pt"

python pred_obb.py --model $MODEL --images_root $IMAGES_ROOT --labels_root $PREDS_ROOT
python cal_map.py --images_root $IMAGES_ROOT --labels_root $LABELS_ROOT --preds_root $PREDS_ROOT
# echo "Start post process"
# for eps in $(seq 0.01 0.01 0.1); do
#     for min_samples in $(seq 1 1 10); do
#         for t_d in $(seq 0.01 0.01 0.1); do
#             for base_conf in $(seq 0.1 0.1 1.0); do
#                 python post_process.py --images_root $IMAGES_ROOT --labels_root $PREDS_ROOT --processed_root $PROCESSED_PREDS_ROOT \
#                     --eps $eps --min_samples $min_samples --t_d $t_d --base_conf $base_conf
#                 echo "post process with eps: $eps min_samples: $min_samples t_d: $t_d base_conf: $base_conf"
#                 python cal_map.py --images_root $IMAGES_ROOT --labels_root $LABELS_ROOT --preds_root $PROCESSED_PREDS_ROOT
#             done
#         done
#     done
# done
# echo "end process"
python post_process.py --images_root $IMAGES_ROOT --labels_root $PREDS_ROOT --processed_root $PROCESSED_PREDS_ROOT --vis
python cal_map.py --images_root $IMAGES_ROOT --labels_root $LABELS_ROOT --preds_root $PREDS_ROOT