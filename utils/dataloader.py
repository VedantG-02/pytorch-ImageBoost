from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import os


class MySuperResolutionDataset(Dataset):
    def __init__(self, root_dir, set='train', test_set=None, transform=None):
        '''
            Args:
            - root_dir (str) : path to the root directory containing train/test data
            - set (str) : 'train'/'test' to access corresponding train/test images; default = 'train'
            - test_set (str) : determine which data to test on : Set5 or Set14 and get results
            - transform : data transformations
        '''
        super(MySuperResolutionDataset, self).__init__()
        self.root_dir = root_dir
        self.set = set
        self.transform = transform
        self.set_dir = os.path.join(root_dir, set) # access the train/test dir
        self.data = []

        # access training images
        if set == 'train':
            self.class_names = os.listdir(self.set_dir) # I've used 2 datasets: DIV2K and Flickr2K; thus, #(self.class_names) = 2
            for id, class_name in enumerate(self.class_names):
                img_dir = os.path.join(self.set_dir, class_name)
                filenames = os.listdir(img_dir)
                for filename in filenames:
                    self.data.append((filename, id)) 

         # access testing images
        elif set == 'test':
            self.test_set_dir = os.path.join(self.set_dir, test_set)
            lr_filenames = []
            hr_filenames = []
            for filename in os.listdir(self.test_set_dir):
                if 'LR' in filename:
                    lr_filenames.append(filename)
                elif 'HR' in filename:
                    hr_filenames.append(filename)
            
            assert len(lr_filenames) == len(hr_filenames), "LR img count doesn't match HR img count"
            
            for _ in range(len(lr_filenames)):
                self.data.append((lr_filenames[_], hr_filenames[_]))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.set == 'train':
            filename, id = self.data[index]
            orig_img = Image.open(os.path.join(os.path.join(self.set_dir, self.class_names[id]), filename))    
            if self.transform:
                cropped_img = self.transform[0](orig_img)
                lr = self.transform[1](cropped_img)      # resize and convert to [0, 1]
                hr = self.transform[2](cropped_img)      # convert to [0, 1]; not according to paper
                return lr, hr
        
        elif self.set == 'test':
            lr_filename, hr_filename = self.data[index]
            lr = Image.open(os.path.join(self.test_set_dir, lr_filename))
            lr = self.convert_8bit_to_24bit(lr)
            hr = Image.open(os.path.join(self.test_set_dir, hr_filename))
            hr = self.convert_8bit_to_24bit(hr)
            if self.transform:
                lr = self.transform(lr)                  # convert to [0, 1]
                hr = self.transform(hr)                  # convert to [0, 1]
                return lr, hr

    def convert_8bit_to_24bit(self, image):
        # convert grayscale (8-bit) image to RGB (24-bit) 
        # 18 examples in Set14 are grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
# --- # --- # --- # --- # --- # --- # --- # --- # --- # --- # --- # --- # 

def test():
    traindata = MySuperResolutionDataset('data', set='train')
    print(f"\nTraining Data:\nlen(traindata): {len(traindata)}")
    print(traindata[0]) # gives None; as transform=None

    testdata5 = MySuperResolutionDataset('data', set='test', test_set='Set5')
    print(f"\nTesting Data (Set5):\nlen(testdata): {len(testdata5)}")
    print(testdata5[0])# gives None; as transform=None

    testdata14 = MySuperResolutionDataset('data', set='test', test_set='Set14')
    print(f"\nTesting Data (Set14):\nlen(testdata): {len(testdata14)}")
    print(testdata14[0])# gives None; as transform=None

    print("\n# ---Testing Done--- #\n")
    

if __name__ == '__main__':
    test()