IMAGES_ROOT="/root/yolo-sar/datasets/competition-2/images"
LABELS_ROOT="/root/yolo-sar/datasets/competition-2/preds"
PROCESSED_LABELS_ROOT="/root/yolo-sar/datasets/competition-2/processed_preds"

MODEL="/root/yolo-sar/runs/obb/train2/weights/best.pt"

python pred_obb.py --model $MODEL --images_root $IMAGES_ROOT --labels_root $LABELS_ROOT
python post_process.py --images_root $IMAGES_ROOT --labels_root $LABELS_ROOT --processed_labels_root $PROCESSED_LABELS_ROOT