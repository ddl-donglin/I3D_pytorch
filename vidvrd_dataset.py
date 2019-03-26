import json
import os
import os.path

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
from tqdm import tqdm


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


def load_rgb_frames(frame_path, begin, end):
    frames = []

    for i in range(begin + 1, end):
        img_path = os.path.join(frame_path, str(i).zfill(4) + '.jpg')
        if not os.path.exists(img_path):
            print(img_path)
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]
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


def make_vidvrd_dataset(anno_rpath, frames_rpath):
    vidvrd_dataset_list = []

    anno_list = os.listdir(anno_rpath)

    with open('dataset/actions.json', 'r') as in_f:
        actions_dict = json.load(in_f)

    pbar = tqdm(total=len(anno_list))
    for each_anno in anno_list:
        with open(os.path.join(anno_rpath, each_anno), 'r') as in_f:
            each_anno_json = json.load(in_f)

        for each_ins in each_anno_json['relation_instances']:
            start_f = each_ins['begin_fid']
            end_f = each_ins['end_fid']
            action_label = each_ins['predicate'].split('_')[0]
            if action_label in actions_dict.keys():
                label = np.full((1, end_f - start_f), actions_dict[action_label])
                frame_path = os.path.join(frames_rpath, each_anno[:-5])
                vidvrd_dataset_list.append((frame_path, label, start_f, end_f))
        pbar.update(1)

    pbar.close()
    return vidvrd_dataset_list


class VidvrdPytorchTrain(data_utl.Dataset):

    def __init__(self, anno_rpath,
                 frames_rpath, mode, save_dir, transforms=None):
        self.data = make_vidvrd_dataset(
            anno_rpath=anno_rpath,
            frames_rpath=frames_rpath)
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

        frame_path, label, start_f, end_f = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(frame_path, start_f, end_f)
        else:
            print(self.mode, 'is not supported')

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)


class VidvrdPytorchExtract(data_utl.Dataset):
    def __init__(self, anno_rpath, save_dir,
                 frames_rpath, mode, transforms=None):
        self.data = make_vidvrd_dataset(
            anno_rpath=anno_rpath,
            frames_rpath=frames_rpath)
        self.frames_rpath = frames_rpath
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

        frame_path, label, start_f, end_f = self.data[index]

        if os.path.exists(frame_path + '.npy'):
            return 0, 0, frame_path

        if os.path.exists(frame_path):
            if self.mode == 'rgb':
                imgs = load_rgb_frames(frame_path, start_f, end_f)
            else:
                print('not supported')

            imgs = self.transforms(imgs)
            return video_to_tensor(imgs), torch.from_numpy(label), frame_path

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    from torchvision import transforms
    import videotransforms

    anno_rpath = '/home/daivd/PycharmProjects/VidVRD-dataset/train_anno'
    frames_rpath = '/home/daivd/PycharmProjects/VidVRD-dataset/frames'
    mode = 'rgb'
    batch_size = 1
    load_model = 'models/rgb_charades.pt'
    save_dir = 'output/features/'
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])

    dataset = VidvrdPytorchExtract(anno_rpath=anno_rpath,
                                   save_dir=save_dir,
                                   frames_rpath=frames_rpath,
                                   mode=mode, transforms=train_transforms)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36,
                                             pin_memory=True)

    for data in dataloader:
        # get the inputs
        inputs, labels, frame_path = data
        if not os.path.exists(frame_path[0]):
            print(frame_path[0])
