import torch 
import torch.nn as nn


# building blocks first:
# conv block
# residual block
# upsample block

class convBlock(nn.Module):
    def __init__(self, in_c, out_c, use_act=True, use_bn=True, disc=False, **kwargs):
        '''
            Args:
            - in_c (int) : input image channels 
            - out_c (int) : convBlock output channels
            - use_act (bool) : flag to include activation in the block
            - use_bn (bool) : flag to inlcude batchnorm in the block
            - disc (bool) : flag to identify if the block is used in discriminator or generator (alters activation)
        '''
        super(convBlock, self).__init__()
        self.use_act = use_act

        self.conv = nn.Conv2d(in_c, out_c, bias=not use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_c) if use_bn==True else nn.Identity()
        self.act = nn.PReLU(num_parameters=out_c) if disc==False else nn.LeakyReLU(0.2)
    
    def forward(self, x):
        if self.use_act:
            return self.act(self.bn(self.conv(x)))
        return self.bn(self.conv(x))
    

class upsampleBlock(nn.Module):
    def __init__(self, in_c, sf):
        '''
            Args:
            - in_c (int) : input channels to the upsample block
            - sf (int) : upscaling factor (for spatial dims of input maps; #channels doesn't change)
        '''
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, in_c*(sf*sf), kernel_size=3, stride=1, padding=1)
        self.pixshuff = nn.PixelShuffle(sf) # out_c == in_c
        # self.deconv = nn.ConvTranspose2d(in_c, in_c, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.pixshuff(self.conv(x)))
        # return self.act(self.deconv(x))


class residualBlock(nn.Module):
    def __init__(self, in_c):
        '''
            Args:
            - in_c (int) : input channels to the residual block
        '''
        super(residualBlock, self).__init__()
        self.conv_block1 = convBlock(in_c, in_c, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = convBlock(in_c, in_c, kernel_size=3, stride=1, padding=1, use_act=False)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        return out + x
    

# gan generator + discriminator
class Generator(nn.Module):
    def __init__(self, sf, in_c=3, num_c=64, B=16):
        '''
            Args:
            - sf (int) : upscaling factor; required for upsample block
            - in_c (int) : input image channels of LR training images; default 3 - RGB images
            - num_c (int) : number of channels each convBlock produce inside generator (same for all)
            - B (int) : number of residual blocks
        '''
        super(Generator, self).__init__()
        self.initial = convBlock(in_c, num_c, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[residualBlock(num_c) for _ in range(B)])
        self.conv_block = convBlock(num_c, num_c, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(*[upsampleBlock(num_c, sf) for _ in range(2)])
        self.final = convBlock(num_c, in_c, kernel_size=9, stride=1, padding=4, use_act=False, use_bn=False)
        self.act = nn.Tanh()

    def forward(self, x):
        initial = self.initial(x)
        out = self.residuals(initial)
        out = self.conv_block(out) + initial
        out = self.upsamples(out)
        return self.act(self.final(out))
    

class Discriminator(nn.Module):
    # input size dim while training (not optimized for any shape input): 3 X 96 X 96
    def __init__(self, in_c=3):
        '''
            Args:
            - in_c (int) : input image channels of HR training images; default 3 - RGB images
        '''
        super(Discriminator, self).__init__()
        channels = [64, 64, 128, 128, 256, 256, 512, 512]
        blocks = []
        
        for id, out_c in enumerate(channels):
            blocks.append(convBlock(
                in_c = in_c,
                out_c = out_c,
                kernel_size = 3,
                stride = 1 + id%2,
                padding = 1,
                use_act = True,
                use_bn = False if id==0 else True,
                disc = True
            ))
            in_c = out_c
        
        self.conv_layers = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.conv_layers(x))

# --- # --- # --- # --- # --- # --- # --- # --- # --- # --- # --- # --- #

def test():
    sf = 2
    imH = 24
    imW = 24

    gen = Generator(sf).cuda()
    x = torch.rand((3, 3, imH, imW)).cuda()
    out_gen = gen(x)
    # out_gen = out_gen.squeeze(0) 
    # print(out_gen.shape)
    assert out_gen.shape == (3, 3, imH*(sf**2), imW*(sf**2)), "gen output shape doesn't match"

    disc = Discriminator(in_c=3).cuda()
    y = torch.rand((3, 3, 96, 96)).cuda()
    out_disc = disc(y)
    # print(out_disc.shape)
    assert out_disc.shape == (3, 1), "disc output shape doesn't match"


    print("\n# ---Testing Done--- #\n")


if __name__ == '__main__':
    test()