import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")
global kernel_sizes


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        # global kernel_sizes
        self.kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")
        if self.mode == 'train':
            self.random_size = random.randint(256, 256)
        else:
            self.random_size = 256

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index, extend=True):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            img_folder = os.path.join(self.prefix, "features/fullFrame-210x260px/" + fi['folder'])
        elif self.dataset == 'CSL':
            img_folder = os.path.join(self.prefix, "features/fullFrame-210x260px/" + fi['folder'] + "/*.jpg")
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])
        img_list = sorted(glob.glob(img_folder))

        # if self.mode == 'train':
        #     img_list = video_augmentation.imageListTemporalRescale(img_list, 0.2, self.frame_interval)

        # img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        # 如果img_list大于128, 随机选取128帧
        # if len(img_list) > 128:
        #     frame_indices = list(range(len(img_list)))
        #     selected_indices = sorted(random.sample(frame_indices, 128))
        #     img_list = [img_list[i] for i in selected_indices]
        # # 如果img_list小于64, 随机重复帧
        # elif len(img_list) < 64:
        #     try:
        #         frame_indices = list(range(len(img_list)))
        #         selected_indices = sorted(random.sample(frame_indices, 64))
        #         frame_indices.extend(selected_indices)
        #         frame_indices.sort()
        #         img_list = [img_list[i] for i in frame_indices]
        #     except Exception as e:
        #         img_list = img_list

        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
            else:
                raise Exception(f"Label {phase} not found in dictionary.")
        # return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi
        resized_img_list = []
        for img_path in img_list:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.random_size, self.random_size), interpolation=cv2.INTER_LANCZOS4)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_img_list.append(img)

        # 反转整个图片序列和文本序列
        # if self.mode == 'train' and random.random() < 0.2:
        #     resized_img_list.reverse()
        #     label_list.reverse()

        # 拼接另一个视频
        # if self.mode == 'train' and extend and random.random() < 0.2:
        #     other_img_list, other_label_list, other_fi = self.read_video(random.randint(0, len(self)-1), extend=False)
        #     if len(other_label_list) + len(label_list) < 26:
        #         resized_img_list.extend(other_img_list)
        #         label_list.extend(other_label_list)

        return resized_img_list, label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                # video_augmentation.RandomVerticalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                # video_augmentation.RandomRotation(30),
                # video_augmentation.RandomAugment(),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
                # video_augmentation.RandomRotate(45),
                # video_augmentation.RandomErase(),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                # video_augmentation.RandomCrop(self.input_size),
                video_augmentation.CenterCrop(self.input_size),
                # video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    # @staticmethod
    def collate_fn(self, batch):
        if self.mode == 'train':
            self.random_size = random.randint(256, 256)
        else:
            self.random_size = 256

        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        
        left_pad = 0
        last_stride = 1
        total_stride = 1
        # global kernel_sizes
        # for layer_idx, ks in enumerate(self.kernel_sizes):
        #     if ks[0] == 'K':
        #         left_pad = left_pad * last_stride
        #         left_pad += int((int(ks[1])-1)/2)
        #     elif ks[0] == 'P':
        #         last_stride = int(ks[1])
        #         total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    # vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                    torch.zeros(max_len - len(vid) - left_pad, vid.shape[1], vid.shape[2], vid.shape[3]),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
