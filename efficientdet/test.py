"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
import pandas as pd
compound_coef = 4
force_input_size = None  # set None to use default size
img_path = '/opt/ml/efficientdet_d4/datasets/test/'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.1  #0.2
iou_threshold = 0.7  #0.2
'''
threshold는 postprocessing에서 classification score가 
이 값보다 큰 object detection 결과를 선택하는 기준값을 의미합니다. 
위 코드에서는 0.05로 설정되어 있습니다.
iou_threshold는 postprocessing에서 non-maximum suppression을 수행할 때, 
겹치는 bounding box들 중 가장 높은 score를 가진 bounding box를 선택하는 기준값을 의미합니다. 
위 코드에서는 0.8로 설정되어 있습니다.

일반적으로 threshold 값은 예측된 박스와 해당 클래스의 예측값의 신뢰도(confidence)를 나타내며, 
이 값이 작을수록 더 많은 박스를 유지합니다. 높은 threshold 값은 정확도가 높지만 재현율이 낮아져서
작은 객체들을 검출하기 어렵게 만들 수 있습니다. 일반적으로 0.1 ~ 0.3 사이의 값으로 설정됩니다.
iou_threshold는 겹치는 예측 박스를 어느 정도 허용할 것인지를 결정하는 값입니다. 
높은 iou_threshold 값은 겹치는 예측 박스를 거의 허용하지 않고 정확도가 높아지지만, 
더 작은 객체들을 검출하지 못하게 됩니다. 일반적으로 0.5 ~ 0.8 사이의 값으로 설정됩니다.
'''
use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

def mk_output(n, preds, imgs):  # 사진 1장당 bbox, score, class 정보
    prediction=''
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(float)
            obj = preds[i]['class_ids'][j]
            score = preds[i]['scores'][j]  # float()
            prediction_string = str(obj) + ' ' + str(score) + ' ' + str(round(x1,5)) + ' ' + str(round(y1,5)) + ' ' + str(round(x2,5)) + ' ' + str(round(y2,5)) + ' '
            prediction+=prediction_string
    return prediction


print('start inference...')
t1 = time.time()

prediction_strings=[]
file_names=[]
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load('/opt/ml/efficientdet/logs/coco/[pth파일 경로 넣기]', map_location='cpu'))
model.requires_grad_(False)
model.eval()
for n in range(len(os.listdir(img_path))):
    img_no = str(n).zfill(4)+'.jpg'
    file_names.append('test/'+img_no)
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path+img_no, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
    out = invert_affine(framed_metas, out)
    prediction_strings.append(mk_output(n, out, ori_imgs))
    
    if n%100==0:
        print(f'[{n//100}/{len(os.listdir(img_path))//100}] is done...')


submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv(os.path.join('/opt/ml/efficientdet_d4/results/',f'submission.csv'), index=None)



t2 = time.time()
tact_time = (t2 - t1) / 10
print(f'{tact_time} seconds, end inference...')
