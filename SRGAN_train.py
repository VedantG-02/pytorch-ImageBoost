import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from loss.loss import MSELoss, VGGLoss
from utils.dataloader import MySuperResolutionDataset
from utils.model import Generator, Discriminator
from utils.utils import saveModel
import argparse
import time
import math


# torch.manual_seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_scratch', const='n', nargs='?', type=str, help='from scratch training (y/n)')
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

    # generator
    if args.from_scratch == 'n':
        gen = Generator(in_c=3, num_c=64, sf=args.sf).to(DEVICE)
        # load MSE based srresnet model
        checkpoint = torch.load('models/SRResNet_MSE')
        gen.load_state_dict(checkpoint['model_state_dict'])
        optimizer_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.999))
        optimizer_gen.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        gen = Generator(in_c=3, num_c=64, sf=args.sf).to(DEVICE)
        optimizer_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # discriminator
    disc = Discriminator(in_c=3).to(DEVICE)
    optimizer_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.999))


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
        content_loss = MSELoss()
    elif args.loss_fn == 'VGG':
        path = r'models/VGG19_pretrained.pth'
        content_loss = VGGLoss(path, i=args.i, j=args.j)
    bce_loss = nn.BCELoss()

    # training
    gen.train()
    disc.train()
    start_time = time.time()
    for epoch in range(args.epochs):
   
        for batch, (lr_image, hr_image) in enumerate(dataloader):
            # send data to gpu/cpu
            lr_image = lr_image.to(DEVICE)
            hr_image = hr_image.to(DEVICE)

            ### train discriminator: max(log(D(x)) + log(1 - D(G(z))))
            fake = gen(lr_image)
            disc_real = disc(hr_image).view(-1)
            disc_fake = disc(fake.detach()).view(-1)
            disc_loss_real = bce_loss(disc_real, torch.ones_like(disc_real)-0.1*torch.rand_like(disc_real))
            disc_loss_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake)+0.1*torch.rand_like(disc_fake))
            loss_disc = (disc_loss_fake + disc_loss_real)/2

            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # train generator: max(log(D(G(z)))
            disc_fake = disc(fake).view(-1)
            adversarial_loss = 1e-3 * bce_loss(disc_fake, torch.ones_like(disc_fake)-0.1*torch.rand_like(disc_fake))
            content_loss_ = content_loss(fake, hr_image)
            loss_gen = content_loss_ + adversarial_loss

            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()  

            # printing loss and mean disc outputs
            if (batch+1) % (len(dataloader)//2) == 0:
                len_epoch = math.floor(math.log10(args.epochs) + 1)
                len_loader = math.floor(math.log10(len(dataloader)) + 1)
                print(f"Epoch [{epoch+1:>{len_epoch}}/{args.epochs}], Batch [{batch+1:>{len_loader}}/{len(dataloader)}], Loss D: {loss_disc:.5f}, loss G: {loss_gen:.5f}")
                print(f"    - Mean discriminator output of the REAL batch : {disc_real.mean().item()}")
                print(f"    - Mean discriminator output of the FAKE batch : {disc_fake.mean().item()}")
        print(f"\n    [ time elapsed: {(time.time()-start_time)/60:.5f} minutes ]\n")


    # model saving
    model_save_dir = 'models' 
    if args.loss_fn == 'MSE':
        PATH = f'{model_save_dir}/SRGAN_MSE'
    elif args.loss_fn == 'VGG':
        PATH = f'{model_save_dir}/SRGAN_VGG{args.i}{args.j}'
    saveModel(gen, optimizer_gen, PATH+'_gen')
    saveModel(disc, optimizer_disc, PATH+'_disc')