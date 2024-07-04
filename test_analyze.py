import sys
sys.path.append('..')

import numpy as np
import os
import csv
import copy
import logging
from tqdm import tqdm
import math
import random
import argparse
import cv2
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TestDataset
from utils import  performances_cross_db, compute_video_score, performances_cross_db_th
from model import MixStyleResCausalModel

def run_test(test_csv, args, device):

    test_dataset = TestDataset(csv_file=test_csv, input_shape=args.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = torch.nn.DataParallel(MixStyleResCausalModel(model_name=args.model_name,  pretrained=False, num_classes=2, ms_layers=[]))
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    save_score = False

    AUC_value, HTER_value = test_model(model, test_loader, device, video_format=args.video_format, save_path=os.path.split(test_csv)[0])

    print(f'Results: AUC= {AUC_value:.4f}, HTER= {HTER_value:.4f} \n')

def test_model(model, data_loader, device, save_path, video_format=True, save_scores=True):

    raw_test_scores, gt_labels = [], []
    raw_scores_dict = []
    raw_test_video_ids = []
    image_paths = []
    fp_df = pd.DataFrame([], columns=['path', 'width', 'height'])
    fn_df = pd.DataFrame([], columns=['path', 'width', 'height'])

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            raw, labels, img_pathes = data["images"].to(device), data["labels"], data["img_path"]
            output = model(raw, cf=None)

            raw_scores = output.softmax(dim=1)[:, 1].cpu().data.numpy()
            raw_test_scores.extend(raw_scores)
            image_paths.extend(img_pathes)
            gt_labels.extend(labels.data.numpy())

            for j in range(raw.shape[0]):
                image_name = os.path.splitext(os.path.basename(img_pathes[j]))[0]
                video_id = os.path.join(os.path.dirname(img_pathes[j]), image_name.rsplit('_', 1)[0])
                raw_test_video_ids.append(video_id)

        if video_format:
            raw_test_scores, gt_labels, _ = compute_video_score(raw_test_video_ids, raw_test_scores, gt_labels)

        raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
        raw_test_scores = ( raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]
        AUC_value, _, _, HTER_value, best_th = performances_cross_db_th(raw_test_scores, gt_labels)
        test_scores = (raw_test_scores > best_th).astype(int)
        
        for idx, (pred, true) in enumerate(zip(test_scores, gt_labels)):
            if pred == 1 and true == 0:
                os.makedirs(f'{save_path}/error_attack/', exist_ok=True)
                img = cv2.imread(image_paths[idx])
                height, width, channels = img.shape
                save_image_path = f'{save_path}/error_attack/{os.path.basename(image_paths[idx])}'
                fp_df = pd.concat([pd.DataFrame([[save_image_path, width, height]], columns=fp_df.columns), fp_df], ignore_index=True)
                cv2.imwrite(save_image_path, img)
            elif pred == 0 and true == 1:
                os.makedirs(f'{save_path}/error_real/', exist_ok=True)
                img = cv2.imread(image_paths[idx])
                height, width, channels = img.shape
                save_image_path = f'{save_path}/error_real/{os.path.basename(image_paths[idx])}'
                fn_df = pd.concat([pd.DataFrame([[save_image_path, width, height]], columns=fn_df.columns), fn_df], ignore_index=True)
    
    fp_df.to_csv(f'{save_path}/error_attack/info.csv', index=False)
    fn_df.to_csv(f'{save_path}/error_real/info.csv', index=False)
    
    return AUC_value, HTER_value


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

    parser.add_argument("--input_shape_width", default=256, type=int, help="Neural Network input shape")
    parser.add_argument("--input_shape_height", default=256, type=int, help="Neural Network input shape")
    parser.add_argument("--batch_size", default=128, type=int, help="train batch size")

    ########## argument should be noted
    parser.add_argument("--model_path", default='checkpoints/best_model.pth', type=str, help="path to saved weights")
    parser.add_argument("--test_csv", type=str, help="csv contains test data")
    parser.add_argument("--video_format", type=bool, help="video format or not")

    args = parser.parse_args()

    print(f"TEST DATA: {args.test_csv} \n  Backbone: {args.model_name},  model_path: {args.model_path},  bs: {args.batch_size} \n")
    print("---------------------------------------")

    args.input_shape = (args.input_shape_width, args.input_shape_height)

    run_test(test_csv=args.test_csv,
                 args=args,
                 device=device)
