#TODO add mutliclass Dice & Rand Loss
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import unet_6
import unet
import skimage.io as io
# from sklearn.metrics import adjusted_rand_score
from torchvision import transforms
from skimage import io
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random


class Fire(data.Dataset):
    def __init__(self, train: bool, root="../data/"):        
        normalize = transforms.Normalize([0.0016779501327332674], [0.5])
        self.transform1 = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform2 = transforms.ToTensor()
        if train:
            self.num = #
            self.root = root + "train/"
        else:
            self.num = #
            self.root = root + "test/"

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        img = self.transform1(io.imread(self.root + "MaxFRP/" + str(index) + ".jpg"))
        mask = self.transform2(io.imread(self.root + "FireMask/" + str(index) + ".jpg")) 
        return (img, mask)

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))
    return alpha
"""
def DiceLoss(a,b):
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1 - ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

def RandLoss(a,b):
    a = (a>=0.5).float()
    a = a.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()
    c = adjusted_rand_score(a,b)
    c = (c+1)/2
    return 1-c
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-pre", metavar="PRE", type=str, default=None, dest="pre")
    parser.add_argument("-lr", metavar="LR", type=float, default=0.001, dest="lr")
    parser.add_argument("-eps", metavar="E", type=int, default=400, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=5e-4, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    parser.add_argument("-sam", metavar="SAM", type=bool, default=False, dest="sam")
    parser.add_argument("-ver", metavar="V", type=int, default=1, dest="ver")
    args = parser.parse_args()

    trainset = Fire(True)
    testset = Fire(False)
    
    trainloader = data.DataLoader(trainset, batch_size=30, shuffle=True, num_workers=32)
    testloader = data.DataLoader(testset, batch_size=30, shuffle=False, num_workers=32)

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    net = unet.run_cnn() if args.ver == 1 else unet_6.run_cnn()
    net = nn.DataParallel(net)
    vall = False
    if args.pre is not None:
        checkpoint = torch.load(args.pre)
        net.load_state_dict(checkpoint["net"])
        vall = True #only for non-GAP pretrained
    net.to(device)

    best_loss = checkpoint["loss"] if vall else 100
    alpha = checkpoint["alpha"] if vall else args.lr
    if args.opt == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=alpha, weight_decay=args.wd)
    elif args.opt == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    if vall:
        optimizer.load_state_dict(checkpoint["optimizer"])
    train_loss, val_loss, test_loss = [], [], []
    train_loss1 = []
    start_ = checkpoint["epoch"] if vall else 1 
    epochs = checkpoint["epoch"]+args.eps if vall else args.eps
    for epoch in range(start_, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        net = net.train()
        epoch_loss, epoch_loss1 = 0.0, 0.0
        cnt = 0
        for img, mask in trainloader:
            mask_type = torch.float32
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            mask_pred = net(img)
            loss = criterion(mask_pred, mask)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
        train_loss.append(epoch_loss/cnt)
        print("Epoch" + str(epoch) + " Train BCE Loss:", epoch_loss/cnt)
        
        net = net.eval()
        tot_val1 = 0.0
        cnt = 0
        with torch.no_grad():
            for img, mask in testloader:
                mask_type = torch.float32
                img, mask = (img.to(device), mask.to(device, dtype=mask_type))
                mask_pred = net(img)
                mask_pred.to(device)
                tot_val1 += criterion(mask_pred, mask).item()
                cnt += 1
        loss_ = tot_val1/cnt
        print("Epoch" + str(epoch) + " Test BCE Loss:", tot_val1/cnt)
        if loss_ < best_loss:
            valid = True
            print("New best test loss!")
            best_loss = loss_
        else:
            valid = False
        val_loss.append(tot_val1/cnt)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":net.state_dict(),
                "loss": loss_,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            path_ = "./models/" + args.opt + "_lr" + str(args.lr) if args.ver == 1 else "./models_6/" + args.opt + "_lr" + str(args.lr) + "/"
            try:
                os.mkdir(path_)
            except:
                pass
            torch.save(state, str(path_ + "best.pt"))
        if epoch == epochs:
            fig = plt.figure()
            plt.plot(train_loss, label="Train")
            plt.plot(val_loss, label="Test")
            plt.xlabel("Epochs")
            plt.ylabel("BCE Loss")
            plt.title("Train-Test BCE Loss")
            fig.savefig(path_+ "BCE.png")
            """
            fig = plt.figure()
            plt.plot(train_loss1, label="Train")
            plt.plot(test_loss, label="Test")
            plt.xlabel("Epochs")
            plt.ylabel("Dice Loss")
            plt.title("Train-Test Dice Loss")
            fig.savefig(path_ + "Dice.png")
            """
            print("Saved plots")
