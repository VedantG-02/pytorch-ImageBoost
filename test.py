import torch
import torch.nn as nn
from torchvision import transforms
from utils.model import Generator
from utils.dataloader import MySuperResolutionDataset
from utils.utils import PSNR, saveImage
import argparse
import math
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_name', type=str, help='path to generator model')
    parser.add_argument('--test_set', type=str, help='Set14 or Set5 set to test on')
    args = parser.parse_args()


    gen = Generator(sf=2)
    checkpoint = torch.load(f'models/{args.gen_name}')
    gen.load_state_dict(checkpoint['model_state_dict'])
    gen.eval().cuda()


    transform_ = transforms.ToTensor()
    dataset = MySuperResolutionDataset(root_dir='data', set='test', test_set=args.test_set, transform=transform_)


    # output saving dirs
    if not os.path.isdir('Results'):
        os.mkdir('Results')
    if not os.path.isdir(f'Results/{args.gen_name}'):
        os.mkdir(f'Results/{args.gen_name}')
    if not os.path.isdir(f'Results/{args.gen_name}/{args.test_set}'):
        os.mkdir(f'Results/{args.gen_name}/{args.test_set}')

    save_dir = f'Results/{args.gen_name}/{args.test_set}'


    psnr = []
    for name_id, (lr, hr) in enumerate(dataset):
        lr = lr.cuda()
        hr = hr.cuda()
        output = gen(lr.unsqueeze(0)) # input requires a 4d tensor; add a dummy dimension
        output = output.squeeze(0)    # convert to 3d tensor; remove the dummy dimension
        psnr.append(PSNR(output, hr))

        saveImage(lr, output, hr, save_dir, name_id)


    # printing avg psnr value
    print(f"\n   [ Average PSNR value of {args.gen_name} model on {args.test_set}: {sum(psnr)/len(psnr)} ]\n")