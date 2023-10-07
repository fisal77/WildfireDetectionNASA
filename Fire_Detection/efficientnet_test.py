import torch
import torch.nn as nn
import cv2
import skimage.io as io
#from efficientnet.models.efficientnet import EfficientNet, params
import argparse
import mlconfig
from efficientnet import models

#def numel(model):
   # return sum(p.numel() for p in model.parameters())


def test_output(model):
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        frame = cv2.resize(frame, (224,224))
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        with torch.no_grad():
            _ = torch.from_numpy(frame).float()
            _ = torch.reshape(_, (1, 3, 224, 224))
            y = model(_.cuda())
            print(torch.sigmoid(y))

def test(model):
    img = io.imread("./img2.jpg")
    img = cv2.resize(img, (224,224))
    img = torch.from_numpy(img).float()
    img = torch.reshape(img, (1, 3, 224, 224))
    print(torch.sigmoid(model(img.cuda())))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="efficientnet_b0")
    parser.add_argument("--resume", type=str, default="experiments/fire/best.pth") #path
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    model = getattr(models, args.arch)(pretrained=args.resume)
    model = nn.DataParallel(model)
    state_dict = torch.load(args.resume, map_location="cuda:%d" % args.gpu)
    #print(state_dict["model"]["classifier.1.bias"].shape)
    model.load_state_dict(state_dict["model"])
    model.to("cuda:%d" % args.gpu)
    model.eval()
    #test(model)
    test_output(model)
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
