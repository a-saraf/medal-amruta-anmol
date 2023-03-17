import torch.nn as nn
import torch

class unet(nn.Module):
    
    def __init__(self, ):
        super(unet, self).__init__()
        
        self.elu=nn.ELU()
        self.sig=nn.Sigmoid()
        self.maxpool = nn.MaxPool3d((2, 2, 2))
        #maxpool and elu
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        #going down
        
        self.conv9 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #bottom
        
        #self.convT1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2,2,3), stride=2, padding=0)
        self.convT2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.convT3 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(2,2,3), stride=2)
        self.convT4 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(2,2,3), stride=2)
        #coming up

        self.conv13 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv14 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv15 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv16 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv17 = nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv18 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        #coming up
        self.conv19 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
       
        #last layer leftttt

        self.dropout1 = nn.Dropout3d(0.1)
        self.dropout2 = nn.Dropout3d(0.2)
        self.dropout3 = nn.Dropout3d(0.3)
        #dropout
        
    def forward(self, x):
        #print (torch.Tensor.size(x))
        out1 = self.elu(self.conv2(self.dropout1(self.elu(self.conv1(x)))))
        #print (torch.Tensor.size(out1))
        out2 = self.maxpool(out1)
        #print (torch.Tensor.size(out2))
        out3 = self.elu(self.conv4(self.dropout1(self.elu(self.conv3(out2)))))
        #print (torch.Tensor.size(out3))
        out4 = self.maxpool(out3)
        #print (torch.Tensor.size(out4))
        out5 = self.elu(self.conv6(self.dropout2(self.elu(self.conv5(out4)))))
        #print (torch.Tensor.size(out5))
        out6 = self.maxpool(out5)
        #print (torch.Tensor.size(out6))

        #going down
                                                
        out9 = self.elu(self.conv10(self.dropout3(self.elu(self.conv9(out6)))))
        #print (torch.Tensor.size(out9))
        #lowermost block
                                                                       
        out13 = self.convT2(out9)
        #print (torch.Tensor.size(out13))
        out14 = torch.cat((out13,out5),1)
        #print (torch.Tensor.size(out14))
        out15 = self.elu(self.conv14(self.dropout2(self.elu(self.conv13(out14)))))
        #print (torch.Tensor.size(out15))
        out16 = self.convT3(out15)
        #print (torch.Tensor.size(out16))
        out17 = torch.cat((out16,out3),1)
        #print (torch.Tensor.size(out17))
        out18 = self.elu(self.conv16(self.dropout1(self.elu(self.conv15(out17)))))
        #print (torch.Tensor.size(out18))
        out19 = self.convT4(out18)
        #print (torch.Tensor.size(out19))
        out20 = torch.cat((out19,out1),1)
        #print (torch.Tensor.size(out20))
        out21 = self.elu(self.conv18(self.dropout1(self.elu(self.conv17(out20)))))
        #print (torch.Tensor.size(out21))
                                                  
        out22 = self.sig(self.conv19(out21))
        #print (torch.Tensor.size(out22))
        
        return out22
