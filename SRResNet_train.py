import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from loss.loss import MSELoss, VGGLoss
from utils.dataloader import MySuperResolutionDataset
from utils.model import Generator
from utils.utils import saveModel
import argparse
import time
import math


# torch.manual_seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--lr', type=float, help='learning rate for optimizer')
    parser.add_argument('--batch_size', const=16, nargs='?', type=int)
    parser.add_argument('--lr_img_size', const=24, nargs='?', type=int, help='lr image size; default 24 as given in paper')
    parser.add_argument('--sf', const=2, nargs='?', type=int, help='scale factor; model will scale each img dim of size sf*sf')
    parser.add_argument('--loss_fn', type=str, help='loss function to be used to pretrain: VGG or MSE (all uppercase)')
    parser.add_argument('--i', type=int, help='PerceptualLoss hyperparameter; either 5 or 2')
    parser.add_argument('--j', type=int, help='PerceptualLoss hyperparameter; either 4 or 2')
    args = parser.parse_args()


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        print("\nUsing GPU\n")
    else:
        print("\nUsing CPU\n")


    SRResNet = Generator(in_c=3, num_c=64, sf=args.sf).to(DEVICE)
    optimizer = optim.Adam(SRResNet.parameters(), lr=args.lr, betas=(0.9, 0.999))


    # transformations
    common_transforms = transforms.Compose([
        transforms.RandomCrop((args.lr_img_size*(args.sf**2), args.lr_img_size*(args.sf**2))),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(degrees=45)
    ])
    lr_transforms = transforms.Compose([
        transforms.Resize((args.lr_img_size, args.lr_img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    hr_transforms = transforms.Compose([
        transforms.ToTensor() 
    ])


    # dataset and dataloader
    transforms_ = [common_transforms, lr_transforms, hr_transforms]
    dataset = MySuperResolutionDataset(root_dir='data', set='train', transform=transforms_)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    # losses
    if args.loss_fn == 'MSE':
        criterion = MSELoss()
    elif args.loss_fn == 'VGG':
        path = r'models/VGG19_pretrained.pth'
        criterion = VGGLoss(path, i=args.i, j=args.j)


    # training
    SRResNet.train()
    start_time = time.time()
    for epoch in range(args.epochs):
        running_loss = 0.0
        
        for batch, (lr_image, hr_image) in enumerate(dataloader):
            # send data to gpu/cpu
            lr_image = lr_image.to(DEVICE)
            hr_image = hr_image.to(DEVICE)
            gen_image = SRResNet(lr_image)
            loss = criterion(gen_image, hr_image)
            running_loss += loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # printing loss 
            if (batch+1) % (len(dataloader)//2) == 0:
                loss = loss.item() 
                len_epoch = math.floor(math.log10(args.epochs) + 1)
                len_loader = math.floor(math.log10(len(dataloader)) + 1)
                print(f"Epoch [{epoch+1:>{len_epoch}}/{args.epochs}], Batch [{batch+1:>{len_loader}}/{len(dataloader)}], Batch loss: {loss:>8f}")
        print(f"\n    [ average loss per image in epoch {epoch+1}: {running_loss/len(dataloader):.8f} ]")
        print(f"    [ time elapsed: {(time.time()-start_time)/60:.5f} minutes ]\n")


    # model saving
    model_save_dir = 'models' 
    if args.loss_fn == 'MSE':
        PATH = f'{model_save_dir}/SRResNet_MSE'
    elif args.loss_fn == 'VGG':
        PATH = f'{model_save_dir}/SRResNet_VGG{args.i}{args.j}'
    saveModel(SRResNet, optimizer, PATH)

    

    