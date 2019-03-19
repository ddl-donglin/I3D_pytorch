import ast
import os

import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from vidor_dataset import VidorPytorchExtract as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def run(anno_rpath, video_rpath, train_split=True, val_split=True,
        frames_rpath='data/Vidor_rgb/JPEGImages/', mode='rgb', batch_size=1,
        load_model='models/rgb_charades.pt', save_dir='output/features/', low_memory=True):

    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

    dataloaders = dict()
    datasets = dict()
    phases = list()

    if train_split:
        dataset = Dataset(anno_rpath=anno_rpath,
                          splits=['training'],
                          video_rpath=video_rpath,
                          frames_rpath=frames_rpath,
                          mode=mode,
                          transforms=train_transforms,
                          low_memory=low_memory,
                          save_dir=save_dir)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36,
                                                 pin_memory=True)

        dataloaders['train'] = dataloader
        datasets['train'] = dataset
        phases.append('train')

    if val_split:
        val_dataset = Dataset(anno_rpath=anno_rpath,
                              splits=['validation'],
                              video_rpath=video_rpath,
                              frames_rpath=frames_rpath,
                              mode=mode,
                              transforms=test_transforms,
                              low_memory=low_memory,
                              save_dir=save_dir)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36,
                                                     pin_memory=True)

        dataloaders['val'] = val_dataloader
        datasets['val'] = val_dataset
        phases.append('val')

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in phases:
        i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0

        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, vid_dir, vidid = data
            npy_save_dir = os.path.join(save_dir, vid_dir[0])
            npy_path = os.path.join(npy_save_dir, vidid[0] + '.npy')

            if not os.path.exists(npy_save_dir):
                # create the directory
                os.mkdir(npy_save_dir)

            if os.path.exists(npy_path):
                continue

            b, c, t, h, w = inputs.shape
            if t > 800:
                features = []
                for start in range(1, t - 56, 800):
                    end = min(t - 1, start + 800 + 56)
                    start = max(1, start - 48)
                    with torch.no_grad():
                        ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda())
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
                np.save(npy_path, np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                with torch.no_grad():
                    inputs = Variable(inputs.cuda())
                features = i3d.extract_features(inputs)
                np.save(npy_path, features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-anno_rpath', type=str, required=True, help='the root path of annotations')
    parser.add_argument('-video_rpath', type=str, required=True, help='the root path of videos')
    parser.add_argument('-frame_rpath', type=str, help='the root path of frame')
    parser.add_argument('-gpu', type=str, default="0", help='gpu_id')
    parser.add_argument('-train_split', type=ast.literal_eval, default=True, help='train_split')
    parser.add_argument('-val_split', type=ast.literal_eval, default=True, help='val_split')
    parser.add_argument('-load_model', type=str)
    parser.add_argument('-save_dirs', type=str)

    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run(args.anno_rpath, args.video_rpath,
        train_split=args.train_split, val_split=args.val_split)
