import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = '/home/shuoxing/XQK/UAV-DETR-main/dataset/dataset_visdrone/data.yaml'
    check_path(yaml_path)
    # model = RTDETR('ultralytics/cfg/models/uavdetr-r50.yaml')
    model = RTDETR('/home/shuoxing/XQK/UAV-DETR-main/ultralytics/cfg/models/uavdetr-r18.yaml')
    model.train(data=str(yaml_path),
                cache=False,
                imgsz=640,
                epochs=500,
                batch=4,
                workers=8,
                device='0',
                #resume='/home/shuoxing/XQK/UAV-DETR-main/runs/train/visdrone_train_SMFF_improve_r18/weights/last.pt', # last.pt path
                pretrained='/home/shuoxing/XQK/UAV-DETR-main/runs/train/visdrone_train_SMFF_improve_r18/weights/best.pt',
                project='runs/train',
                name='visdrone_train_SMFF_improveFloat_r18',
                patience = 20 # early stopping patience
                )