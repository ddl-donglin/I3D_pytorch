import argparse
import os

import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

import videotransforms
from pytorch_i3d import InceptionI3d
from vidvrd_dataset import VidvrdPytorchExtract as Dataset


def run(anno_rpath, frames_rpath, mode='rgb', batch_size=1,
        load_model='models/rgb_charades.pt', save_dir='output/features/'):
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass

    dataset = Dataset(anno_rpath=anno_rpath,
                      save_dir=save_dir,
                      frames_rpath=frames_rpath,
                      mode=mode, transforms=train_transforms)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36,
                                             pin_memory=True)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
        # i3d = nn.DataParallel(i3d, device_ids=device_ids)

    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    # Iterate over data.

    print('Begin 2 extract features: ')
    pbar = tqdm(total=800)
    for data in dataloader:
        # get the inputs
        inputs, labels, frame_path = data
        vid_id = frame_path[0].split('/')[-1]

        print('Now is extracting: ', vid_id)
        pbar.update(1)

        npy_save_dir = os.path.join(save_dir, vid_id)
        npy_path = os.path.join(npy_save_dir, vid_id + '.npy')

        if not os.path.exists(npy_save_dir):
            # create the directory
            os.mkdir(npy_save_dir)

        if os.path.exists(npy_path):
            continue

        b, c, t, h, w = inputs.shape
        if t > 400:
            features = []
            for start in range(1, t - 56, 400):
                end = min(t - 1, start + 400 + 56)
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

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-anno_rpath', type=str, required=True, help='the root path of annotations')
    parser.add_argument('-frame_rpath', type=str, help='the root path of frame')
    parser.add_argument('-gpu', type=str, default="0", help='gpu_id')
    parser.add_argument('-load_model', type=str)
    parser.add_argument('-save_dir', type=str)

    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run(args.anno_rpath, args.frame_rpath)
