from PIL import Image
import os

dir = r'data\test\Set14'
count = 0
for filename in os.listdir(dir):
    if filename.endswith('.png'):
        path = os.path.join(dir, filename)
        img = Image.open(path)
        if img.mode != 'RGB':
            count += 1
            print(img, filename)
print(f"\nThere are {count} images in Set14 testset which are not RGB\n")


dir = r'data\test\Set5'
count = 0
for filename in os.listdir(dir):
    if filename.endswith('.png'):
        path = os.path.join(dir, filename)
        img = Image.open(path)
        if img.mode != 'RGB':
            count += 1
            print(img, filename)
print(f"There are {count} images in Set5 testset which are not RGB\n")


dir = r'data\train\Flickr2K'
count = 0
for _ in os.listdir(dir):
    path = os.path.join(dir, _)
    img = Image.open(path)
    if img.mode != 'RGB':
        count += 1
        print(img, _)
print(f"There are {count} images in Flickr2K trainset which are not RGB\n")


dir = r'data\train\DIV2K_train'
count = 0
for _ in os.listdir(dir):
    path = os.path.join(dir, _)
    img = Image.open(path)
    if img.mode != 'RGB':
        count += 1
        print(img, _)
print(f"There are {count} images in DIV2K_train trainset which are not RGB\n")
