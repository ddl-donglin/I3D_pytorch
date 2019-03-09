import json
import os
import os.path
from collections import defaultdict
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
from tqdm import tqdm
from frames import extract_all_frames
from dataset.vidor import VidOR


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(video_path, begin, end):
    image_dir = 'data/frames'
    frames = []
    extract_all_frames(video_path, image_dir)

    for i in range(begin, end):
        img = cv2.imread(os.path.join(image_dir, str(i).zfill(4) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_vidor_dataset(anno_rpath, splits, video_rpath, task, low_memory=True):
    vidor_dataset = VidOR(anno_rpath, video_rpath, splits, low_memory)
    if task not in ['object', 'action', 'relation']:
        print(task, "is not supported! ")
        exit()

    vidor_dataset_list = []
    if task == 'action':
        actions = [
            'watch', 'bite', 'kiss', 'lick', 'smell', 'caress', 'knock', 'pat',
            'point_to', 'squeeze', 'hold', 'press', 'touch', 'hit', 'kick',
            'lift', 'throw', 'wave', 'carry', 'grab', 'release', 'pull',
            'push', 'hug', 'lean_on', 'ride', 'chase', 'get_on', 'get_off',
            'hold_hand_of', 'shake_hand_with', 'wave_hand_to', 'speak_to', 'shout_at', 'feed',
            'open', 'close', 'use', 'cut', 'clean', 'drive', 'play(instrument)',
        ]

        for each_split in splits:
            for ind in vidor_dataset.get_index(each_split):
                for each_ins in vidor_dataset.get_action_insts(ind):
                    video_path = vidor_dataset.get_video_path(ind)
                    start_f, end_f = each_ins['duration']
                    label = np.full((1, end_f - start_f + 1), actions.index(each_ins['category']))
                    vidor_dataset_list.append((video_path, label, start_f, end_f))

    return vidor_dataset_list


class VidorPytorchTrain(data_utl.Dataset):

    def __init__(self, anno_rpath, splits, video_rpath, mode, task='action', transforms=None, low_memory=True):
        self.data = make_vidor_dataset(anno_rpath, video_rpath, splits, task, low_memory)
        self.splits = splits
        self.transforms = transforms
        self.mode = mode
        self.task = task

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        video_path, label, start_f, end_f = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(video_path, start_f, end_f)
        else:
            # imgs = load_flow_frames(self.root, vid, start_f, 64)
            print('not supported')
        label = label[:, start_f: end_f]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)


class VidorPytorchExtract(data_utl.Dataset):
    def __init__(self, anno_rpath, save_dir, splits, video_rpath, mode, task='action', transforms=None, low_memory=True):
        self.data = make_vidor_dataset(anno_rpath, video_rpath, splits, task, low_memory)
        self.splits = splits
        self.transforms = transforms
        self.mode = mode
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        video_path, label, start_f, end_f = self.data[index]
        vid_paths = video_path.split('/')

        if os.path.exists(os.path.join(self.save_dir, vid_paths[-2], vid_paths[-1][:-4] + '.npy')):
            return 0, 0, vid_paths[-2], vid_paths[-1][:-4]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(video_path, start_f, end_f)
        else:
            # imgs = load_flow_frames(self.root, vid, start_f, 64)
            print('not supported')

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid_paths[-2], vid_paths[-1][:-4]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    base_path = '/home/daivd/PycharmProjects/vidor/'
    anno_rpath = base_path + 'annotation'
    # splits = ['training', 'validation']
    splits = ['validation']
    video_rpath = base_path + 'val_vids'
    task = 'action'
    vidor_dataset_list = make_vidor_dataset(anno_rpath, splits, video_rpath, task)
    print(vidor_dataset_list)
