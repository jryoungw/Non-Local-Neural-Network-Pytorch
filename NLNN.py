import torch
from torch import nn
import torch.nn.functional as F

class NLNN(nn.Module):
    def __init__(self, kernel_size=1, in_channels=1024, instance='embedded_gaussian', dim=2, shrink_factor=2):
        super(NLNN, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = in_channels // shrink_factor
        self.shrink_factor = shrink_factor
        self.instance = instance
        self.dim = dim
        assert self.dim in [1,2,3]
        if self.dim == 1:
            self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv3 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv4 = nn.Conv1d(self.out_channels, self.in_channels, self.kernel_size)
            self.concatConv = nn.Conv1d(self.in_channels, 1, self.kernel_size)
        elif self.dim == 2:
            self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv4 = nn.Conv2d(self.out_channels, self.in_channels, self.kernel_size)
            self.concatConv = nn.Conv2d(self.in_channels, 1, self.kernel_size)
        else:
            self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv2 = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv3 = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size)
            self.conv4 = nn.Conv3d(self.out_channels, self.in_channels, self.kernel_size)
            self.concatConv = nn.Conv3d(self.in_channels, 1, self.kernel_size)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
    def forward(self, x):
        assert self.instance in ['gaussian', 'embedded_gaussian', 'dot_product']
        assert len(x.size()) in [3,4,5]
        assert len(x.size()) == self.dim+2, print("Dimension of x should be same compared to 'dim' argument in NLNN class. Write NLNN(dim=len(x.shape))")
        if self.instance == 'embedded_gaussian':
            x_theta = self.conv1(x)
            x_phi = self.conv2(x)
            x_g = self.conv3(x)
            x_g = x_g.view(x.size()[0], -1, self.out_channels)

            x_theta = x_theta.view(x.size()[0], -1, self.out_channels)
            x_phi = x_phi.view(x.size()[0], self.out_channels, -1)
            x_inter = torch.matmul(x_theta, x_phi)
            x_inter = self.softmax(x_inter, axis=0)

            x_inter = torch.matmul(x_inter, x_g)
            x_inter = x_inter.view(x.size()[0], x.size()[1]//self.shrink_factor, *x.size()[2:])
            x_last = self.conv4(x_inter)

            return x+x_last

        elif self.instance == 'gaussian':
            x_theta = x.clone()
            x_phi = x.clone()
            x_g = self.conv3(x)
            x_g = x_g.view(x.size()[0], -1, self.out_channels)

            x_theta = x_theta.view(x.size()[0], -1, self.in_channels)
            x_phi = x_phi.view(x.size()[0], self.in_channels, -1)
            x_inter = torch.matmul(x_theta, x_phi)
            x_inter = self.softmax(x_inter, axis=0)
            x_inter = torch.matmul(x_inter, x_g)
            x_inter = x_inter.view(x.size()[0], x.size()[1]//self.shrink_factor, *x.size()[2:])
            x_last = self.conv4(x_inter)

            return x+x_last
        elif self.instance == 'dot_product':
            x_theta = self.conv1(x)
            x_phi = self.conv2(x)
            x_g = self.conv3(x)
            x_g = x_g.view(x.size()[0], -1, self.out_channels)

            x_theta = x_theta.view(x.size()[0], -1, self.out_channels)
            x_phi = x_phi.view(x.size()[0], self.out_channels, -1)
            x_inter = torch.matmul(x_theta, x_phi)
            x_inter = x_inter / x_inter.size(-1)

            x_inter = torch.matmul(x_inter, x_g)
            x_inter = x_inter.view(x.size()[0], x.size()[1]//self.shrink_factor, *x.size()[2:])
            x_last = self.conv4(x_inter)

            return x+x_last

        
if __name__ == '__main__':
    
    for mode in ['gaussian', 'embedded_gaussian', 'dot_product']:
        print('+'*10+' instance : {} '.format(mode)+'+'*10)
        print('1-dimensional NLNN')
        tensor = torch.rand((1,512,128))
        model = NLNN(in_channels=tensor.size()[1], instance=mode, dim=1)
        output = model(tensor)
        print('2-dimensional NLNN')
        tensor = torch.rand((1,512,64,64))
        model = NLNN(in_channels=tensor.size()[1], instance=mode, dim=2)
        output = model(tensor)
        print('3-dimensional NLNN')
        tensor = torch.rand((1,16,4,4,4))
        model = NLNN(in_channels=tensor.size()[1], instance=mode, dim=3)
        output = model(tensor)

