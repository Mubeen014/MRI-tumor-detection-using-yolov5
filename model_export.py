import os
import shutil

base_path = ''
model_path = os.path.join(base_path, 'yolov5/runs/train/exp/weights/best.pt')
shutil.copy(model_path, base_path)
