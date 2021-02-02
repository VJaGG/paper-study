import torch
import numpy as np
from fcn import FCN
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
import torch.optim as optim
from dataset import VOCdataset
import torch.utils.data as data
from tensorboardX import SummaryWriter
from utils import AverageMeter, ProgressMeter, Evaluator


def train(train_loader, model, criteria, optimizer, device, batch_size):
    model.train()
    evaluator = Evaluator(21)
    evaluator.reset()
    train_loss = AverageMeter("Loss", ":.4")
    progress = ProgressMeter(len(train_loader), train_loss)
    for i, (image, mask) in enumerate(train_loader):
        image = image.to(device)
        mask = mask.to(device)

        output = model(image)
        loss = criteria(output, mask)

        predict = output.data.cpu().numpy()
        predict = np.argmax(predict, axis=1)
        target = mask.cpu().numpy()
        evaluator.add_batch(target, predict)
        train_loss.update(loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            progress.print(i)
            evaluator.add_batch(target, predict)
    info = {"loss": train_loss.val,
            "pixel acc": evaluator.Pixel_Accuracy(),
            "mean acc": evaluator.Pixel_Accuracy_Class(),
            "miou": evaluator.Mean_Intersection_over_Union()}
    return info


def validate(val_loader, model, criteria, device, batch_size):
    model.eval()
    evaluator = Evaluator(21)
    evaluator.reset()
    val_loss = []
    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(val_loader)):
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = criteria(output, mask)
            predict = output.data.cpu().numpy()
            predict = np.argmax(predict, axis=1)
            target = mask.cpu().numpy()
            evaluator.add_batch(target, predict)
            val_loss.append(loss.item())
    info = {"loss": sum(val_loss) / len(val_loader),
            "pixel acc": evaluator.Pixel_Accuracy(),
            "mean acc": evaluator.Pixel_Accuracy_Class(),
            "miou": evaluator.Mean_Intersection_over_Union()}
    return info


def main():
    root = "./data/VOCdevkit/VOC2012"
    batch_size = 4
    num_workers = 4
    num_classes = 21
    lr = 0.0025
    # lr = 5e-4  # fine-tune
    epoches = 100
    writer = SummaryWriter(comment="-fcn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = A.Compose([
                                 A.HorizontalFlip(),  # 注意这个先后顺序
                                 A.VerticalFlip(),
                                #  A.transpose(p=0.5),
                                 A.RandomRotate90(),
                                #  A.ElasticTransform(p=1, alpha=120,
                                #                     sigma=120 * 0.05,
                                #                     alpha_affine=120 * 0.03),
                                A.RandomResizedCrop(320, 480),
                                ])
    val_transform = A.Compose([
        A.RandomResizedCrop(320, 480)])
    train_set = VOCdataset(root, mode="train", transform=train_transform)
    val_set = VOCdataset(root, mode="val", transform=val_transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)
    val_loader = data.DataLoader(val_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    model = FCN(num_classes).to(device)
    # state_dict = torch.load("./model/best.pth")
    # print("loading pretrained parameters")
    # model.load_state_dict(state_dict)
    # del state_dict
    criteria = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
    #                       weight_decay=2e-4)

    vgg_parameters = (list(map(id, model.encode1.parameters()))+
                      list(map(id, model.encode2.parameters()))+
                      list(map(id, model.encode3.parameters()))+
                      list(map(id, model.encode4.parameters()))+
                      list(map(id, model.encode5.parameters())))
    encode_parameters = (list(model.encode1.parameters())+
                         list(model.encode2.parameters())+
                         list(model.encode3.parameters())+
                         list(model.encode4.parameters())+
                         list(model.encode5.parameters()))

    decode_parameters = filter(lambda p: id(p) not in vgg_parameters, model.parameters())
    optimizer = optim.SGD([{'params': encode_parameters, 'lr': 0.1 * lr},
                           {'params': decode_parameters, 'lr': lr}],
                          momentum=0.9,
                          weight_decay=2e-3)

    # optimizer = optim.Adam([{'params': encode_parameters, 'lr': 0.1 * lr},
    #                         {'params': decode_parameters, 'lr': lr}],
    #                        weight_decay=2e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.85)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                            T_0=100,
    #                                                            T_mult=1,
    #                                                            eta_min=0.0001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_miou = 0.0
    for epoch in range(1, epoches+1):
        print("Epoch: ", epoch)
        scheduler.step()
        train_info = train(train_loader, model,
                           criteria, optimizer, device, batch_size)
        val_info = validate(val_loader, model,
                            criteria, device, batch_size)
        string = "loss: {}, pixel acc: {}, mean acc: {} miou: {}"
        print("train", end=' ')
        print(string.format(train_info['loss'],
                            train_info["pixel acc"],
                            train_info['mean acc'],
                            train_info['miou']))
        print("val", end=' ')
        print(string.format(val_info['loss'],
                            val_info['pixel acc'],
                            val_info['mean acc'],
                            val_info['miou']))

        writer.add_scalar("lr",
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          epoch)
        writer.add_scalar('train/loss', train_info['loss'], epoch)
        writer.add_scalar('train/pixel acc', train_info['pixel acc'], epoch)
        writer.add_scalar('train/mean acc', train_info['mean acc'], epoch)
        writer.add_scalar('train/miou', train_info['miou'], epoch)
        writer.add_scalar('val/loss', val_info['loss'], epoch)
        writer.add_scalar('val/pixel acc', val_info['pixel acc'], epoch)
        writer.add_scalar('val/mean acc', val_info['mean acc'], epoch)
        writer.add_scalar('val/miou', val_info['miou'], epoch)
        if val_info['miou'] > best_miou:
            best_miou = val_info['miou']
            torch.save(model.state_dict(), './model/best.pth')
            print("best model find at {} epoch".format(epoch))


if __name__ == "__main__":
    main()
