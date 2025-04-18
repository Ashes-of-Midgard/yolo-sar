import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--preds_root', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    dummy_model = YOLO("/root/models/yolo12x-wtconv-SRSDD-competition-aug-add-epochs300.pt", task="obb")
    dummy_model.val(data=args.dataset, pred_txts=args.preds_root, batch=1)
    # dummy_model.val(data=args.dataset, batch=1)
    

if __name__ == '__main__':
    main()