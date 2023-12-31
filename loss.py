import torch
import torch.nn as nn
import torch.nn.functional as F

class L_TV(nn.Module): # Total variation loss
    def __init__(self):
        super(L_TV,self).__init__()

    def forward(self,x):
        bs_x, c_x, h_x, w_x = x.size()
        tv_h = torch.pow(x[:,:,1:,:]-x[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,:,1:]-x[:,:,:,:-1], 2).sum()
        return (tv_h+tv_w)/(bs_x*c_x*h_x*w_x)
    

class L_exp(nn.Module):
    def __init__(self,patch_size=16):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    def forward(self, x, mean_val ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([mean_val] ).cuda(),2))
        return d

class L_LMCV(nn.Module): # local maximum color value loss
    def __init__(self, LMCV_value=1, max_patch_factor=8):
        super(L_LMCV,self).__init__()
        self.LMCV_value = LMCV_value
        self.max_patch_factor = max_patch_factor

    def forward(self,x):
        max_patch_size = x.shape[-1]//self.max_patch_factor
        pad_x = F.pad(input = x, 
                    pad = ((max_patch_size-1)//2,(max_patch_size)//2,(max_patch_size-1)//2,(max_patch_size)//2), 
                    mode = 'replicate')
        
        x_LMCV = F.max_pool2d(pad_x, 
                    kernel_size = max_patch_size, 
                    stride = 1,
                    padding = 0,
                    dilation = 1,
                    return_indices = False,
                    ceil_mode = False
                    )
        x_LMCV,_ = torch.max(x_LMCV, dim=1)
        x_LMCV = x_LMCV.mean()
        return torch.abs(self.LMCV_value-x_LMCV)
    
class L_DARK(nn.Module): # dark channel loss
    def __init__(self, DARK_value=1, max_patch_factor=8):
        super(L_DARK,self).__init__()
        self.DARK_value = DARK_value
        self.max_patch_factor = max_patch_factor

    def forward(self,x):
        # max_patch_size = x.shape[-1]//self.max_patch_factor
        # pad_x = F.pad(input = x, 
        #             pad = ((max_patch_size-1)//2,(max_patch_size)//2,(max_patch_size-1)//2,(max_patch_size)//2), 
        #             mode = 'replicate')
        
        # x_DARK = -F.max_pool2d(-pad_x, 
        #             kernel_size = max_patch_size, 
        #             stride = 1,
        #             padding = 0,
        #             dilation = 1,
        #             return_indices = False,
        #             ceil_mode = False
        #             )
        # x_DARK,_ = torch.min(x_DARK, dim=1)
        # x_DARK = x_DARK.mean()
        # return torch.abs(self.DARK_value-x_DARK)
        x_DARK,_ = torch.min(x, dim=1)
        x_DARK = x_DARK.mean()
        return torch.abs(self.DARK_value-x_DARK)
    
class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k
