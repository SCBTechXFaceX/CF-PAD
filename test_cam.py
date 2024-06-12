import numpy as np
import cv2
import os
import sys
sys.path.append('..')

import numpy as np
import os
import random
import argparse
import albumentations
import torch
from albumentations.pytorch import ToTensorV2

from model import MixStyleResCausalModel

PRE__MEAN = [0.485, 0.456, 0.406]
PRE__STD = [0.229, 0.224, 0.225]
INPUT__FACE__SIZE = 256

def preprocess_frame_pipe():
    return albumentations.Compose([
        albumentations.Resize(height=INPUT__FACE__SIZE, width=INPUT__FACE__SIZE),
        albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
        ToTensorV2(),    
    ])

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = preprocess_frame_pipe()(image=frame)['image']
    frame = torch.tensor(frame).unsqueeze(0)
    return frame

def run_cam_test(args, device):
    model = torch.nn.DataParallel(MixStyleResCausalModel(model_name=args.model_name,  pretrained=False, num_classes=2, ms_layers=[]))
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    cap = cv2.capture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_frame = preprocess_frame(frame).to(device)
            output = model(input_frame, cf=None)
            raw_scores = output.softmax(dim=1)[:, 1].cpu().data.numpy()
            print("frame output:", output)
            print("raw scores: ", raw_scores)
            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # esc key
                break
    cap.release()
    cv2.destroyAllWindows()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')

    set_seed(seed=777)

    parser = argparse.ArgumentParser(description='CF baseline')
    parser.add_argument("--prefix", default='CF', type=str, help="description")
    parser.add_argument("--model_name", default='resnet18', type=str, help="model backbone")

    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")

    ########## argument should be noted
    parser.add_argument("--model_path", default='checkpoints/ocim.pth', type=str, help="path to saved weights")

    args = parser.parse_args()
    run_cam_test(args=args,
                 device=device)
