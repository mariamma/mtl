
import os
import torch
import torchvision
import numpy as np
import cv2
import pandas as pd
from PIL import Image

def segmentation(input_image, unet,device,image_name, fileext,save_path):
    """
        This function segments the lung region from the given input image
        
        inputs - 
            input_image: image input of type Image.PIL
            unet: lung segmentation model
            device: torch.device("cpu") ot torch.device("gpu")
            image_name: name of the image to save the segmented lung region
            filext: extension of the image to save. "png, jpg or jpeg".
            save_path: folder path to save the segmented image.
        
        returns nothing. saves the segmented lung region 
    """
    
    origin = torchvision.transforms.functional.resize(input_image, (512, 512))
    origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
    with torch.no_grad():
        origin = torch.stack([origin])
        origin = origin.to(device)
        out = unet(origin) #logits
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)
    origin = origin[0].to("cpu")
    lung_masked = out[0].to("cpu")
    
    mi = np.array(lung_masked, np.uint8)
    np_image = np.array(input_image.resize((512,512)), np.uint8)
    
    mi = (mi[:, :] * 255.).astype(np.uint8)

    resized = cv2.bitwise_and(np_image, mi)
    resized=resized*255
    resized=resized.astype('uint8')
    return resized


        
def segmentation_func(image_path, segmentation_model, device):
    
    """
        external function used to call the main segmentation function.
        input-
            image_path: path to the image.
            segmentation_model: lung segmentation model
            device: torch.device("cpu") ot torch.device("gpu")
    """
    
    full_path = image_path    
    if 1:
        input_image = Image.open(full_path).convert("P")
        
        trial_save_lung = full_path.rstrip(os.path.basename(full_path))
        
        _, fileext = os.path.splitext(os.path.basename(full_path))
        seg_image_name = 'segmented'
        
        
        resized = segmentation(input_image, segmentation_model, device, seg_image_name, fileext, trial_save_lung)
        
        
        return resized

        

class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out


class PretrainedUNet(torch.nn.Module):
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        self.init_conv = torch.nn.Conv2d(in_channels, 3, 1)

        endcoder = torchvision.models.vgg11(pretrained=True).features
        self.conv1 = endcoder[0]   # 64
        self.conv2 = endcoder[3]   # 128
        self.conv3 = endcoder[6]   # 256
        self.conv3s = endcoder[8]  # 256
        self.conv4 = endcoder[11]   # 512
        self.conv4s = endcoder[13]  # 512
        self.conv5 = endcoder[16]  # 512
        self.conv5s = endcoder[18] # 512

        self.center = Block(512, 512, 256, batch_norm)

        self.dec5 = Block(512 + 256, 512, 256, batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm)

        self.out = torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def forward(self, x): 
        init_conv = torch.nn.functional.relu(self.init_conv(x), inplace=True)

        enc1 = torch.nn.functional.relu(self.conv1(init_conv), inplace=True)
        enc2 = torch.nn.functional.relu(self.conv2(self.down(enc1)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3(self.down(enc2)), inplace=True)
        enc3 = torch.nn.functional.relu(self.conv3s(enc3), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4(self.down(enc3)), inplace=True)
        enc4 = torch.nn.functional.relu(self.conv4s(enc4), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5(self.down(enc4)), inplace=True)
        enc5 = torch.nn.functional.relu(self.conv5s(enc5), inplace=True)

        center = self.center(self.down(enc5))

        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5], 1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))

        out = self.out(dec1)

        return out