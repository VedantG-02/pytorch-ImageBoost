import torch
import torch.nn as nn


def MSELoss():
    return nn.MSELoss()


class VGGLoss(nn.Module):
    '''
        phi_i,j = feature map obtained by the j-th convolution (after activation) before the i-th maxpooling layer
    '''
    def __init__(self, VGG19_path, i=2, j=2):
        '''
            Args:
            - VGG19_path (str) : path to saved vgg19 pretrained model
            - i (int) : 'i' of phi_i,j; default 2
            - j (int) : 'j' of phi_i,j; default 2
        '''
        super(VGGLoss, self).__init__()
        self.i = i
        self.j = j

        self.vgg = torch.load(VGG19_path)
        if i==5 and j==4:
            # after 4th conv + activation layer before 5th maxpool layer
            self.feature_extractor = nn.Sequential(*self.vgg.features[:36]).eval()
        elif i==2 and j==2:
            # after 2nd conv + activation layer before 2nd maxpool layer
            self.feature_extractor = nn.Sequential(*self.vgg.features[:9]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False 
        self.feature_extractor = self.feature_extractor.cuda()  
        self.mse_loss = nn.MSELoss()

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        perceptual_loss = self.mse_loss(sr_features, hr_features)
        return 0.006*perceptual_loss