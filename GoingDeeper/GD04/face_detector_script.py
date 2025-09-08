import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import os, cv2, time
import tqdm
import numpy as np
import math
from itertools import product
import matplotlib.pyplot as plt
from PIL import Image
import io
import pickle

PROJECT_PATH = 'D:/project/Aiffel/face detector'
DATA_PATH = os.path.join(PROJECT_PATH, 'widerface')
MODEL_PATH = os.path.join(PROJECT_PATH, 'checkpoints')
TRAIN_PT_PATH = os.path.join(PROJECT_PATH, 'data', 'train_data.pt')
VALID_PT_PATH = os.path.join(PROJECT_PATH, 'data', 'val_data.pt')
CHECKPOINT_PATH = os.path.join(PROJECT_PATH, 'checkpoints')

# 메모리 오류 수정을 위해 개별 파일들을 저장할 경로
TRAIN_PROCESSED_PATH = os.path.join(PROJECT_PATH, 'data', 'train_processed')
VALID_PROCESSED_PATH = os.path.join(PROJECT_PATH, 'data', 'val_processed')

DATASET_LEN = 12880
BATCH_SIZE = 8
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 256
IMAGE_LABELS = ['background', 'face']

print(torch.__version__)

# =================================================================================================
# 먼저 bounding box 파일을 분석해 봅시다. 분석에 필요한 코드를 함수 형태로 준비할게요.
# bounding box 파일의 한 줄을 입력받아, x0, y0, w, h 값을 반환하는 함수입니다.
def parse_box(data):
    x0 = int(data[0])
    y0 = int(data[1])
    w = int(data[2])
    h = int(data[3])
    return x0, y0, w, h

print('슝=3')

# =================================================================================================
# bounding box 파일 전체를 분석하는 함수입니다.
# 파일 이름과 bounding box 좌표 리스트를 튜플로 묶어 반환합니다.
def parse_widerface(file):
    infos = []
    with open(file) as fp:
        line = fp.readline()
        while line:
            n_object = int(fp.readline())
            boxes = []
            for i in range(n_object):
                box = fp.readline().split(' ')
                x0, y0, w, h = parse_box(box)
                if (w == 0) or (h == 0):
                    continue
                boxes.append([x0, y0, w, h])
            if n_object == 0:
                box = fp.readline().split(' ')
                x0, y0, w, h = parse_box(box)
                boxes.append([x0, y0, w, h])
            infos.append((line.strip(), boxes))
            line = fp.readline()
    return infos

print('슝=3')

# =================================================================================================
# 이미지를 읽고, RGB로 변환한 후, CHW 형태의 Tensor로 바꾸는 함수입니다.
# image_file: 이미지 파일 경로
# return: (에러코드, 이미지 파일의 바이너리 문자열, CHW 형태의 Tensor)
# 에러코드: 0(성공), 1(실패)
# 이미지 파일의 바이너리 문자열: 이미지 파일을 바이너리 모드로 읽은 문자열
# CHW 형태의 Tensor: (채널, 높이, 너비) 형태의 Tensor
# 에러 발생 시, CHW 형태의 Tensor는 None을 반환합니다.
def process_image(image_file):
    try:
        with open(image_file, 'rb') as f:
            image_string = f.read()
            image_data = Image.open(io.BytesIO(image_string)).convert('RGB')
            image_data = torch.from_numpy(np.array(image_data)).permute(2, 0, 1)  # HWC to CHW
            return 0, image_string, image_data
    except Exception as e:
        return 1, image_string, None

print('슝=3')

# =================================================================================================
# bounding box 좌표를 VOC 형식으로 변환하는 함수입니다.
# file_name: 이미지 파일 이름
# boxes: bounding box 좌표 리스트 (x0, y0, w, h)
# image_data: CHW 형태의 Tensor
# return: 이미지 정보 딕셔너리
# 이미지 정보 딕셔너리: {'filename': 파일 이름, 'width': 너비, 'height': 높이, 'depth': 채널 수,
#                     'class': 클래스 리스트, 'xmin': xmin 리스트, 'ymin': ymin 리스트,
#                     'xmax': xmax 리스트, 'ymax': ymax 리스트, 'difficult': difficult 리스트}
# 클래스는 1(얼굴)로 고정, difficult는 0(쉬움)으로 고정
# xmin, ymin, xmax, ymax는 VOC 형식에 맞게 변환
# 에러 발생 시, 빈 딕셔너리를 반환합니다.
# VOC 형식: xmin, ymin, xmax, ymax (왼쪽 위 꼭짓점, 오른쪽 아래 꼭짓점)
# xywh 형식: x0, y0, w, h (왼쪽 위 꼭짓점, 너비, 높이)
def xywh_to_voc(file_name, boxes, image_data):
    shape = image_data.shape
    image_info = {}
    image_info['filename'] = file_name
    image_info['width'] = shape[1]
    image_info['height'] = shape[0]
    image_info['depth'] = 3

    difficult = []
    classes = []
    xmin, ymin, xmax, ymax = [], [], [], []

    for box in boxes:
        classes.append(1)
        difficult.append(0)
        xmin.append(box[0])
        ymin.append(box[1])
        xmax.append(box[0] + box[2])
        ymax.append(box[1] + box[3])
    image_info['class'] = classes
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax
    image_info['difficult'] = difficult

    return image_info

print('슝=3')

# =================================================================================================
# 준비한 함수를 테스트해 봅시다.
# wider_face_split/wider_face_train_bbx_gt.txt 파일을 분석해 봅시다.
# 앞에서 준비한 parse_widerface 함수를 사용합니다.
# bounding box 파일의 앞 5개 이미지만 테스트해 봅니다.
# bounding box 파일의 경로
file_path = os.path.join(DATA_PATH, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
for i, info in enumerate(parse_widerface(file_path)):
    print('--------------------')
    image_file = os.path.join(DATA_PATH, 'WIDER_train', 'images', info[0])
    _, image_string, image_data = process_image(image_file)
    boxes = xywh_to_voc(image_file, info[1], image_data)
    print(boxes)
    if i > 3:
        break

# =================================================================================================
# 이미지 파일의 바이너리 문자열과 이미지 정보를 입력받아, 학습에 사용할 예제 딕셔너리를 반환하는 함수입니다.
# image_string: 이미지 파일의 바이너리 문자열
# image_infos: 이미지 정보 딕셔너리 리스트
# return: 예제 딕셔너리 (이미지 파일 이름, 너비, 높이, 채널 수, 클래스 리스트, xmin 리스트, ymin 리스트,
#                     xmax 리스트, ymax 리스트, 이미지 파일의 바이너리 문자열) 형태 의 딕셔너리 
import io

def make_example(image_string, image_infos):
    for info in image_infos:
        filename = info['filename']
        width = info['width']
        height = info['height']
        depth = info['depth']
        classes = info['class']
        xmin = info['xmin']
        ymin = info['ymin']
        xmax = info['xmax']
        ymax = info['ymax']

    # 이미지 데이터를 numpy 배열로 변환
    image_data = np.frombuffer(image_string, dtype=np.uint8)
    image_data = Image.open(io.BytesIO(image_data)).convert('RGB')
    image_data = np.array(image_data)

    # 데이터를 dict 형태로 저장
    example = {
        'filename': filename,
        'height': height,
        'width': width,
        'classes': classes,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'image_raw': image_data
    }

    return example

print('슝=3')

# =================================================================================================
# 기존 코드 주석 처리
# # 준비한 함수를 사용하여, 학습과 검증에 사용할 데이터셋을 만듭니다.
# # 학습 데이터셋은 wider_face_split/wider_face_train_bbx_gt.txt 파일을 사용하고,
# # 검증 데이터셋은 wider_face_split/wider_face_val_bbx_gt.txt 파일을 사용합니다.
# # 각각의 데이터셋은 .pt 파일로 저장합니다.
# # (2) 데이터셋 생성
# 
# import torch
# import os
# import tqdm
# from PIL import Image
# 
# for split in ['train', 'val']:
#     if split == 'train':
#         output_file = TRAIN_PT_PATH
#         anno_txt = 'wider_face_train_bbx_gt.txt'
#         file_path = 'WIDER_train'
#     else:
#         output_file = VALID_PT_PATH
#         anno_txt = 'wider_face_val_bbx_gt.txt'
#         file_path = 'WIDER_val'
# 
#     dataset = []  # 데이터를 저장할 리스트
# 
#     for info in tqdm.tqdm(parse_widerface(os.path.join(DATA_PATH, 'wider_face_split', anno_txt))):
#         image_file = os.path.join(DATA_PATH, file_path, 'images', info[0])
#         error, image_string, image_data = process_image(image_file)
#         boxes = xywh_to_voc(image_file, info[1], image_data)
# 
#         if not error:
#             example = make_example(image_string, [boxes])
#             dataset.append(example)
# 
#     # dataset을 .pt 파일로 저장
#     torch.save(dataset, output_file)
# 
# print('슝=3')

# 메모리 오류 해결을 위한 수정 코드
import torch
import os
import tqdm
from PIL import Image

for split in ['train', 'val']:
    if split == 'train':
        output_dir = TRAIN_PROCESSED_PATH
        anno_txt = 'wider_face_train_bbx_gt.txt'
        file_path = 'WIDER_train'
    else:
        output_dir = VALID_PROCESSED_PATH
        anno_txt = 'wider_face_val_bbx_gt.txt'
        file_path = 'WIDER_val'

# 개별 파일들을 저장할 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f'{output_dir} 디렉토리를 생성했습니다.')

# tqdm의 enumerate를 사용하여 인덱스와 데이터를 함께 가져옵니다.
    data_source = parse_widerface(os.path.join(DATA_PATH, 'wider_face_split', anno_txt))
    for i, info in enumerate(tqdm.tqdm(data_source)):
        image_file = os.path.join(DATA_PATH, file_path, 'images', info[0])
        error, image_string, image_data = process_image(image_file)
        boxes = xywh_to_voc(image_file, info[1], image_data)

        if not error:
            example = make_example(image_string, [boxes])
# 개별 파일로 즉시 저장
            torch.save(example, os.path.join(output_dir, f'{i}.pt'))

print('슝=3')

# =================================================================================================
!ls /content/drive/MyDrive/data/face_detector/data

# =================================================================================================
# # 파일이 잘 저장되었는지 확인해볼까요??
# # 저장된 파일 로드
# train_data = torch.load(TRAIN_PT_PATH)
# val_data = torch.load(VALID_PT_PATH)
# 
# # 데이터 개수 확인
# print(f"Train 데이터 개수: {len(train_data)}")
# print(f"Validation 데이터 개수: {len(val_data)}")
# 
# # 샘플 데이터 확인
# sample = train_data[0]
# print(f"샘플 데이터 타입: {type(sample)}")
# print(f"샘플 데이터 내용: {sample}")

#   메모리 오류 해결을 위한 수정된 데이터 로딩 코드
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ProcessedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
#   디렉토리에서 .pt 파일 목록을 가져옵니다.
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
#   파일 이름을 정수형으로 변환하여 정렬합니다. (예: '1.pt', '10.pt', '2.pt' -> '1.pt', '2.pt', '10.pt')
        self.file_list.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
#   데이터셋의 전체 길이는 파일의 개수입니다.
        return len(self.file_list)

    def __getitem__(self, idx):
#   주어진 인덱스(idx)에 해당하는 파일을 로드합니다.
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        sample = torch.load(file_path, weights_only=False)  # PyTorch 버전 변경으로 인해 weights_only=False 옵션 추가
        return sample

#   파일이 잘 저장되었는지 확인
#   저장된 개별 파일들을 로드하기 위한 Dataset 객체 생성
train_dataset = ProcessedDataset(TRAIN_PROCESSED_PATH)
val_dataset = ProcessedDataset(VALID_PROCESSED_PATH)

#   데이터 개수 확인
print(f"Train 데이터 개수: {len(train_dataset)}")
print(f"Validation 데이터 개수: {len(val_dataset)}")

#   샘플 데이터 확인 (첫 번째 데이터)
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"샘플 데이터 타입: {type(sample)}")
#   샘플 데이터의 모든 키와 값의 타입을 출력해봅니다.   
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: {type(value)}, shape: {value.shape}")
        else:
            print(f"  - {key}: {type(value)}")
else:
    print("Train 데이터셋이 비어있습니다.")

# DataLoader 사용 예시 (실제 학습 시 이렇게 사용합니다)
#   train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =================================================================================================
BOX_MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
BOX_STEPS = [8, 16, 32, 64]

print('슝=3')

# =================================================================================================
image_sizes = (IMAGE_HEIGHT, IMAGE_WIDTH)
min_sizes = BOX_MIN_SIZES
steps= BOX_STEPS

feature_maps = [
    [math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)]
    for step in steps
]
feature_maps

# =================================================================================================
이제 feature map별로 순회를 하면서 default box 를 생성해 보겠습니다.

# =================================================================================================
boxes = []
for k, f in enumerate(feature_maps):
    for i, j in product(range(f[0]), range(f[1])):
        for min_size in min_sizes[k]:
            s_kx = min_size / image_sizes[1]
            s_ky = min_size / image_sizes[0]
            cx = (j + 0.5) * steps[k] / image_sizes[1]
            cy = (i + 0.5) * steps[k] / image_sizes[0]
            boxes += [cx, cy, s_kx, s_ky]

len(boxes)

# =================================================================================================
pretty_boxes = np.asarray(boxes).reshape([-1, 4])
print(pretty_boxes.shape)
print(pretty_boxes)

# =================================================================================================
def default_box():
    image_sizes = (IMAGE_HEIGHT, IMAGE_WIDTH)
    min_sizes = BOX_MIN_SIZES
    steps= BOX_STEPS
    feature_maps = [
        [math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)]
        for step in steps
    ]
    boxes = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_sizes[1]
                s_ky = min_size / image_sizes[0]
                cx = (j + 0.5) * steps[k] / image_sizes[1]
                cy = (i + 0.5) * steps[k] / image_sizes[0]
                boxes += [cx, cy, s_kx, s_ky]
    boxes = np.asarray(boxes).reshape([-1, 4])
    return boxes

print('슝=3')

# =================================================================================================
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    block_id = id(inputs)

    if strides == (2, 2):
        x = F.pad(inputs, (1, 1, 1, 1), mode='constant', value=0)  # ZeroPadding2D
        x = nn.Conv2d(inputs.size(1), filters, kernel_size=kernel, stride=strides, padding=0, bias=False)(x)
    else:
        x = nn.Conv2d(inputs.size(1), filters, kernel_size=kernel, stride=strides, padding='same', bias=False)(inputs)

    x = nn.BatchNorm2d(filters)(x)
    return F.relu(x)

print('슝=3')

# =================================================================================================
class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1, 1)):
        super(DepthwiseConvBlock, self).__init__()
        self.strides = strides

        if strides != (1, 1):
            self.pad = nn.ZeroPad2d((1, 1, 1, 1))
        else:
            self.pad = nn.Identity()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=strides, padding=0 if strides != (1, 1) else 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.depthwise(x)
        x = F.relu(self.bn1(x))
        x = self.pointwise(x)
        return F.relu(self.bn2(x))

print('슝=3')

# =================================================================================================
class BranchBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(BranchBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, filters * 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x1 = self.conv2(x1)
        x2 = self.conv3(x)
        x = torch.cat([x1, x2], dim=1)
        return F.relu(x)

print('슝=3')

# =================================================================================================
class HeadBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

print('슝=3')

# =================================================================================================
def _compute_heads(inputs, num_class, num_cell):
    conf = HeadBlock(inputs.size(1), num_cell * num_class)(inputs)
    conf = conf.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, num_class)

    loc = HeadBlock(inputs.size(1), num_cell * 4)(inputs)
    loc = loc.permute(0, 2, 3, 1).contiguous().view(inputs.size(0), -1, 4)

    return conf, loc

print('슝=3')

# =================================================================================================
class SsdModel(nn.Module):
    def __init__(self, image_height, image_width, image_labels):
        super(SsdModel, self).__init__()

        self.base_channel = 16
        self.num_cells = [3, 2, 2, 3]
        self.num_class = len(image_labels)

        self.conv_blocks = nn.ModuleList([
            DepthwiseConvBlock(3, self.base_channel * 4, strides=(1, 1)),
            BranchBlock(self.base_channel * 4, self.base_channel * 8)
        ])

        self.layers = nn.ModuleList([
            DepthwiseConvBlock(self.base_channel * 8, self.base_channel * 16, strides=(2, 2)),
            BranchBlock(self.base_channel * 16, self.base_channel),
            DepthwiseConvBlock(self.base_channel * 16, self.base_channel * 16, strides=(2, 2)),
            BranchBlock(self.base_channel * 16, self.base_channel)
        ])

    def forward(self, x):
        x1 = self.conv_blocks[0](x)
        x2 = self.conv_blocks[1](x1)

        x3 = self.layers[0](x2)
        x3 = self.layers[1](x3)

        x4 = self.layers[2](x3)
        x4 = self.layers[3](x4)

        extra_layers = [x1, x2, x3, x4]

        confs, locs = [], []

        for layer, num_cell in zip(extra_layers, self.num_cells):
            conf, loc = _compute_heads(layer, self.num_class, num_cell)
            confs.append(conf)
            locs.append(loc)

        confs = torch.cat(confs, dim=1)
        locs = torch.cat(locs, dim=1)

        predictions = torch.cat([locs, confs], dim=2)

        return predictions

print('슝=3')

# =================================================================================================
model = SsdModel(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_LABELS)
print("the number of model layers: ", len(list(model.modules())))
print(model)

print('슝=3')

# =================================================================================================
# _compute_heads 함수 구현
class _compute_heads(nn.Module):
    def __init__(self, in_channels, num_cell, out_channels):
        super(_compute_heads, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_cell * out_channels, kernel_size=3, padding=1)
        self.num_cell = num_cell
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, self.out_channels)
        return out

class YourSsdModel(nn.Module):
    def __init__(self, num_classes=2, input_shape=(3, 300, 300), num_cells=[4, 6, 6, 6]):
        super(YourSsdModel, self).__init__()

        self.num_classes = num_classes
        self.num_cells = num_cells

        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conf_layers = nn.ModuleList()
        self.loc_layers = nn.ModuleList()

        for num_cell in num_cells:
            self.conf_layers.append(_compute_heads(256, num_cell, self.num_classes))  # 클래스 예측
            self.loc_layers.append(_compute_heads(256, num_cell, 4))  # 바운딩 박스 좌표 예측

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        confs = []
        locs = []

        for conf_layer, loc_layer in zip(self.conf_layers, self.loc_layers):
            confs.append(conf_layer(x))
            locs.append(loc_layer(x))

        confs = torch.cat(confs, dim=1)  # (batch, total_boxes, num_classes)
        locs = torch.cat(locs, dim=1)  # (batch, total_boxes, 4)

        return confs, locs

print('슝=3')

# =================================================================================================
# 모델 불러오기,
mymodel = YourSsdModel()
print(mymodel)

# =================================================================================================
def _crop(img, labels, max_loop=250):
    shape = img.shape

    def matrix_iof(a, b):
        lt = torch.maximum(a[:, None, :2], b[:, :2])
        rb = torch.minimum(a[:, None, 2:], b[:, 2:])

        area_i = torch.prod(rb - lt, dim=2) * (lt < rb).all(dim=2).float()
        area_a = torch.prod(a[:, 2:] - a[:, :2], dim=1)

        return area_i / torch.maximum(area_a[:, None], torch.tensor(1.0))

    for _ in range(max_loop):
        pre_scale = torch.tensor([0.3, 0.45, 0.6, 0.8, 1.0], dtype=torch.float32)
        scale = pre_scale[torch.randint(0, 5, (1,))]

        short_side = min(shape[0], shape[1])
        h = w = int(scale * short_side)

        h_offset = torch.randint(0, shape[0] - h + 1, (1,)).item()
        w_offset = torch.randint(0, shape[1] - w + 1, (1,)).item()

        roi = torch.tensor([w_offset, h_offset, w_offset + w, h_offset + h], dtype=torch.float32)

        value = matrix_iof(labels[:, :4], roi[None, :])
        if torch.any(value >= 1):
            centers = (labels[:, :2] + labels[:, 2:4]) / 2

            mask_a = (roi[:2] < centers).all(dim=1) & (centers < roi[2:]).all(dim=1)
            if mask_a.any():
                img_t = img[h_offset:h_offset + h, w_offset:w_offset + h, :]

                labels_t = labels[mask_a]
                labels_t[:, :4] -= torch.tensor([w_offset, h_offset, w_offset, h_offset], dtype=torch.float32)

                return img_t, labels_t

    return img, labels

print('슝=3')

# =================================================================================================
def _resize(img, labels):
    h_f, w_f = img.shape[1:3]

    locs = torch.stack([labels[:, 0] / w_f, labels[:, 1] / h_f,
                        labels[:, 2] / w_f, labels[:, 3] / h_f], dim=1)

    locs = torch.clamp(locs, 0, 1.0)
    labels = torch.cat([locs, labels[:, 4].unsqueeze(1)], dim=1)

    resize_case = torch.randint(0, 5, (1,)).item()

    resize_methods = [
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.BICUBIC),
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.BOX),
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.NEAREST),
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.LANCZOS3),
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=T.InterpolationMode.BILINEAR)
    ]

    img = resize_methods[resize_case](img)

    return img, labels

print('슝=3')

# =================================================================================================
def _flip(img, labels):
    flip_case = torch.randint(0, 2, (1,)).item()

    if flip_case == 0:
        img = torch.flip(img, dims=[2])

        labels = torch.stack([
            1 - labels[:, 2], labels[:, 1],
            1 - labels[:, 0], labels[:, 3],
            labels[:, 4]
        ], dim=1)

    return img, labels

print('슝=3')

# =================================================================================================
def _pad_to_square(img):
    h, w = img.shape[1:3]

    if h > w:
        pad = (h - w) // 2
        img = F.pad(img, (pad, h - w - pad, 0, 0), value=img.mean())
    elif w > h:
        pad = (w - h) // 2
        img = F.pad(img, (0, 0, pad, w - h - pad), value=img.mean())

    return img

print('슝=3')

# =================================================================================================
def _distort(img):
    img = T.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.1)(img)
    return img

print('슝=3')

# =================================================================================================
def _intersect(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, 0] * inter[:, :, 1]

print('슝=3')

# =================================================================================================
def _jaccard(box_a, box_b):
    inter = _intersect(box_a, box_b)

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter

    return inter / union

print('슝=3')

# =================================================================================================
def _encode_bbox(matched, boxes, variances=[0.1, 0.2]):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - boxes[:, :2]
    g_cxcy /= (variances[0] * boxes[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / boxes[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    g_wh = torch.where(torch.isinf(g_wh), torch.zeros_like(g_wh), g_wh)

    return torch.cat([g_cxcy, g_wh], dim=1)

print('슝=3')

# =================================================================================================
def encode_pt(labels, boxes):
    match_threshold = 0.45

    boxes = boxes.float()
    bbox = labels[:, :4]
    conf = labels[:, -1]

    overlaps = _jaccard(bbox, boxes)

    best_box_overlap, best_box_idx = overlaps.max(dim=1)
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)

    best_truth_overlap[best_box_idx] = 2.0
    best_truth_idx[best_box_idx] = torch.arange(best_box_idx.size(0))

    matches_bbox = bbox[best_truth_idx]
    loc_t = _encode_bbox(matches_bbox, boxes)

    conf_t = conf[best_truth_idx]
    conf_t[best_truth_overlap < match_threshold] = 0

    return torch.cat([loc_t, conf_t.unsqueeze(1)], dim=1)

print('슝=3')

# =================================================================================================
def _transform_data(train, boxes):
    def transform_data(img, labels):
        img = img.float()

        if train:
            img, labels = _crop(img, labels)
            img = _pad_to_square(img)

        img, labels = _resize(img, labels)

        if train:
            img, labels = _flip(img, labels)

        if train:
            img = _distort(img)

        labels = encode_pt(labels, boxes)
        img = img / 255.0

        return img, labels

    return transform_data

print('슝=3')

# =================================================================================================
def _parse_pt(train, boxes):
    def parse_pt(pt):
        example = pickle.loads(pt.numpy())

        img = torch.tensor(np.frombuffer(example['image_raw'], dtype=np.uint8).reshape(3, example['height'], example['width']))

        labels = torch.tensor(np.stack([
            example['x_mins'],
            example['y_mins'],
            example['x_maxes'],
            example['y_maxes'],
            example['classes']], axis=1), dtype=torch.float32)

        img, labels = _transform_data(train, boxes)(img, labels)

        return img, labels

    return parse_pt

# =================================================================================================
def load_pt_dataset(pt_name, train=True, boxes=None, buffer_size=64):
    with open(pt_name, 'rb') as f:
        raw_data = f.read()

    raw_array = np.frombuffer(raw_data, dtype=np.uint8)
    raw_dataset = [torch.tensor(raw_array, dtype=torch.uint8)]

    if train:
        np.random.shuffle(raw_dataset)

    return raw_dataset

print('슝=3')

# =================================================================================================
def load_dataset(boxes, train=True, buffer_size=64):
    pt_name = TRAIN_PT_PATH if train else VALID_PT_PATH

    return load_pt_dataset(pt_name, train, boxes, buffer_size)

print('슝=3')

# =================================================================================================
class PiecewiseConstantWarmUpDecay:
    def __init__(self, boundaries, values, warmup_steps, min_lr):
        if len(boundaries) != len(values) - 1:
            raise ValueError("The length of boundaries should be 1 less than the length of values")

        self.boundaries = boundaries
        self.values = values
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        if step <= self.warmup_steps:
            return self.min_lr + step * (self.values[0] - self.min_lr) / self.warmup_steps

        for i, boundary in enumerate(self.boundaries):
            if step <= boundary:
                return self.values[i]

        return self.values[-1]

print('슝=3')

# =================================================================================================
def MultiStepWarmUpLR(initial_learning_rate, lr_steps, lr_rate, warmup_steps=0, min_lr=0):
    assert warmup_steps <= lr_steps[0]
    assert min_lr <= initial_learning_rate

    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)

    return PiecewiseConstantWarmUpDecay(boundaries=lr_steps, values=lr_steps_value, warmup_steps=warmup_steps, min_lr=min_lr)

print('슝=3')

# =================================================================================================
def hard_negative_mining(loss, class_truth, neg_ratio):
    pos_idx = class_truth > 0
    num_pos = pos_idx.sum(dim=1)
    num_neg = num_pos * neg_ratio

    _, rank = loss.sort(dim=1, descending=True)
    neg_idx = rank < num_neg.unsqueeze(1)

    return pos_idx, neg_idx

print('슝=3')

# =================================================================================================
class MultiBoxLoss:
    def __init__(self, num_classes, neg_pos_ratio=3.0):
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio

    def __call__(self, y_true, y_pred):
        loc_pred, class_pred = y_pred[..., :4], y_pred[..., 4:]
        loc_truth, class_truth = y_true[..., :4], y_true[..., 4].long()

        temp_loss = F.cross_entropy(class_pred, class_truth, reduction='none')
        pos_idx, neg_idx = hard_negative_mining(temp_loss, class_truth, self.neg_pos_ratio)

        loss_class = F.cross_entropy(class_pred[pos_idx | neg_idx], class_truth[pos_idx | neg_idx], reduction='sum')

        loss_loc = F.smooth_l1_loss(loc_pred[pos_idx], loc_truth[pos_idx], reduction='sum')

        num_pos = pos_idx.float().sum()

        loss_class /= num_pos
        loss_loc /= num_pos

        return loss_loc, loss_class

print('슝=3')

# =================================================================================================
boxes = default_box()
train_dataset = load_dataset(boxes, train=True)

print('슝=3')

# =================================================================================================
model = SsdModel(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_LABELS)
print("the number of model layers: ", len(list(model.modules())))
print(model)

# =================================================================================================
steps_per_epoch = DATASET_LEN // BATCH_SIZE # 한 epoch 당 스텝 수

learning_rate = MultiStepWarmUpLR(
    initial_learning_rate=1e-2,
    lr_steps=[e * steps_per_epoch for e in [50, 70]],
    lr_rate=0.1,
    warmup_steps=5 * steps_per_epoch,
    min_lr=1e-4
)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
multi_loss = MultiBoxLoss(len(IMAGE_LABELS), neg_pos_ratio=3)

print(steps_per_epoch)

# =================================================================================================
def train_step(model, optimizer, criterion, inputs, labels):
    model.train()
    optimizer.zero_grad()

    predictions = model(inputs)
    loss_loc, loss_class = criterion(labels, predictions)

    total_loss = loss_loc + loss_class
    total_loss.backward()

    optimizer.step()

    return total_loss.item(), {'loc': loss_loc.item(), 'class': loss_class.item()}

print('슝=3')

# =================================================================================================
# 학습 루프
EPOCHS = 1

for epoch in range(EPOCHS):
    for step, (inputs, labels) in enumerate(train_dataset):
        load_t0 = time.time()

        total_loss, losses = train_step(model, optimizer, multi_loss, inputs, labels)

        load_t1 = time.time()
        batch_time = load_t1 - load_t0

        print(f"\rEpoch: {epoch + 1}/{EPOCHS} | Batch {step + 1}/{steps_per_epoch} | Batch time {batch_time:.3f} || Loss: {total_loss:.6f} | loc loss:{losses['loc']:.6f} | class loss:{losses['class']:.6f} ", end='', flush=True)

