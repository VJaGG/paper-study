import cv2
import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch.utils.data as data


class VOCdataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.image_set = osp.join(root, 'JPEGImages')
        self.segmentation = osp.join(root, 'SegmentationClass')
        label_path = "label.csv"
        self.col2ind = self.color2index(label_path)
        img_txt = osp.join(root+'/ImageSets/Segmentation', mode+'.txt')
        self.images = []
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        with open(img_txt, 'r') as f:
            info = f.readlines()
            for name in info:
                name = name.strip()
                self.images.append(name)
        print("{} : {}".format(mode, len(self.images)))
        self.transform = transform

    def __getitem__(self, index):
        name = self.images[index]
        image_path = osp.join(self.image_set, name+'.jpg')
        mask_path = osp.join(self.segmentation, name+'.png')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        img = self.img2tensor((img / 255.0 - self.mean) / self.std)
        mask = self.encode2mask(mask)
        mask = torch.from_numpy(mask).long()
        return img, mask

    def __len__(self):
        return len(self.images)

    def img2tensor(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img.astype(np.float32, copy=False))

    def color2index(self, label_path):
        col2ind = np.zeros(256 ** 3)
        color_map = pd.read_csv(label_path, sep=',')
        for i in range(color_map.shape[0]):
            color_info = color_map.iloc[i]
            value = (color_info['r'] * 256 + color_info['g']) * 256 \
                     + color_info['b']
            col2ind[value] = i
        return col2ind

    def encode2mask(self, mask):
        data = np.array(mask, dtype='int32')  # 注意这边的类型转换，不然会为0
        idx = (data[..., 0] * 256 + data[..., 1]) * 256 + data[..., 2]
        return np.array(self.col2ind[idx], dtype='int64')


if __name__ == "__main__":
    root = "./data/VOCdevkit/VOC2012"
    import albumentations as A
    train_transform = A.Compose([
                                 A.HorizontalFlip(),
                                 A.VerticalFlip(),
                                 A.RandomRotate90(),
                                 A.Resize(320, 480),
                                 ])
    train_dataset = VOCdataset(root, mode='train', transform=train_transform)
    # img, mask = train_dataset[0]
    for i in range(len(train_dataset)):
        img, mask = train_dataset[i]
        print(img.shape)
        print(mask.shape)
        assert img.shape == (3, 320, 480)
        assert mask.shape == (320, 480)
        break