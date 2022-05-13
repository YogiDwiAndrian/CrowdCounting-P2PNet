import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--input_vid', default='./data/video/crowd.mp4',
                        help="path where video input")
    parser.add_argument('--output_vid', default='./data/output/crowd_result.mp4',
                        help='path where to save')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help="threshold for object detection")
    parser.add_argument('--mode', default=1, type=int,
                        help="--mode 1: preview output, --mode 2: generate and saving video, --mode 3: preview output and saving video")

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


def resize(frame):
    # load the images
    img_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_raw = Image.fromarray(img_cv)
    return img_raw.resize((720, 576), Image.ANTIALIAS)

def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    # get the P2PNet
    model = build_model(args)
    # GPU
    model = model.cuda()
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your video path here
    vidCap = cv2.VideoCapture(args.input_vid)

    if (vidCap.isOpened() == False):
        print("Error reading video file")

    if args.mode != 1:
        fourcc = 'mp4v'  # output video codec
        fps = vidCap.get(cv2.CAP_PROP_FPS)
        vid_writer = cv2.VideoWriter(args.output_vid, cv2.VideoWriter_fourcc(*fourcc), fps, (720, 576))

    # set cv2
    size = 2
    fontface = cv2.FONT_HERSHEY_DUPLEX
    fontscale = 0.6
    fontcolor = (255, 255, 255)
    start_point = (0, 22)
    end_point = (135, 56)
    while (True):
        ret, frame = vidCap.read()  # reads the next frame

        if ret == True:
            img_resize = resize(frame)
            # pre-proccessing
            img = transform(img_resize).cuda()

            # run inference
            outputs = model(img.unsqueeze(0))
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            outputs_points = outputs['pred_points'][0]

            # filter the predictions
            points = outputs_points[outputs_scores > args.threshold].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > args.threshold).sum())

            # draw circle the predictions
            img_to_draw = cv2.cvtColor(np.array(img_resize), cv2.COLOR_RGB2BGR)
            for p in points:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

            # draw count the predictions
            img_to_draw = cv2.rectangle(img_to_draw, start_point, end_point, (34, 34, 178), -1)
            img_to_draw = cv2.putText(img_to_draw, f"Count : {predict_cnt}", (5, 45), fontface, fontscale, fontcolor)
            img_to_draw = cv2.rectangle(img_to_draw, (420, 2), (1000, 20), (34, 34, 178), -1)
            # img_to_draw = cv2.putText(img_to_draw, f"by SYNNEX METRODATA INDONESIA & DAMAI   ", (430, 15), fontface, 0.4, fontcolor)

            if args.mode == 1:
                # show preview video
                cv2.imshow("video", img_to_draw)
            elif args.mode == 2:
                # generate and saving video
                vid_writer.write(img_to_draw)
            else:
                # show preview video
                cv2.imshow("video", img_to_draw)
                # generate and saving video
                vid_writer.write(img_to_draw)

            if cv2.waitKey(1) == ord('q'):
                break

        else:
            break

    vidCap.release()
    if args.mode != 1:
        vid_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
