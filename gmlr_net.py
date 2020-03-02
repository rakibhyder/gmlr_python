import torch
from torch import nn
import numpy as np
#from torchsummary import summary

def gmlr_net(nz,ngf,ndf,nc,ng,seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if ngf==32:
        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    # input is Z, going into a convolutiontorch.
                    nn.ConvTranspose2d(     nz, ng * 8, 4, 1, 0, bias=False),
        #            nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ng * 8, ng * 4, 4, 2,1, bias=False),
        #            nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ng * 4, ng * 2, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(    ng* 2,      nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 32 x 32
                )
        
            def forward(self, input):
                if input.is_cuda and self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
                else:
                    output = self.main(input)
                return output
            
    elif ngf==64:
        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    # input is Z, going into a convolutiontorch.
                    nn.ConvTranspose2d(     nz, ng * 8, 4, 1, 0, bias=False),
        #            nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ng * 8, ng * 4, 4, 2,1, bias=False),
        #            nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ng * 4, ng * 2, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ng * 2, ng * 1, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 32 x 32
                    nn.ConvTranspose2d(    ng* 1,      nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )
        
            def forward(self, input):
                if input.is_cuda and self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
                else:
                    output = self.main(input)
                return output           
    elif ngf==256:
        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    # input is Z, going into a convolutiontorch.
                    nn.ConvTranspose2d(     nz, ng * 8, 4, 1, 0, bias=False),
        #            nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ng * 8, ng * 4, 4, 2,1, bias=False),
        #            nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ng * 4, ng * 2, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ng * 2, ng * 1, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 32 x 32
                    nn.ConvTranspose2d(ng * 1, ng /2, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 64 x 64
                    nn.ConvTranspose2d(ng / 2, ng /4, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 128 x 128
                    nn.ConvTranspose2d(    ng/4,      nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 256 x 256
                )
        
            def forward(self, input):
                if input.is_cuda and self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
                else:
                    output = self.main(input)
                return output  
    elif ngf==512:
        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    # input is Z, going into a convolutiontorch.
                    nn.ConvTranspose2d(     nz, ng * 8, 4, 1, 0, bias=False),
        #            nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ng * 8, ng * 4, 4, 2,1, bias=False),
        #            nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ng * 4, ng * 2, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ng * 2, ng * 1, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 32 x 32
                    nn.ConvTranspose2d(ng * 1, ng /2, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 64 x 64
                    nn.ConvTranspose2d(ng / 2, ng /4, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 128 x 128
                    nn.ConvTranspose2d(ng / 4, ng /8, 4, 2, 1, bias=False),
        #            nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 256 x 256
                    nn.ConvTranspose2d(    ng/8,      nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 512 x 512
                )
        
            def forward(self, input):
                if input.is_cuda and self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
                else:
                    output = self.main(input)
                return output    
            
    np.random.seed(seed)        
    generator = Generator(1).to(device)
#    summary(generator,(nz,1,1))
    return generator