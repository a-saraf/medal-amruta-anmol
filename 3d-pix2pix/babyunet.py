class unet(nn.Module):
    
    def __init__(self, ):
        super(unet, self).__init__()
        
        self.elu=nn.ELU()
        self.sig=nn.Sigmoid()
        self.maxpool = nn.MaxPool3d((2, 2, 2))
        #maxpool and elu
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv7 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        #going down
        
        self.conv9 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv10 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3)
        #bottom
        
        self.convT1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.convT2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.convT3 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=2)
        self.convT4 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3, stride=2)
        #coming up

        self.conv11 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv12 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv13 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv14 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv15 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv16 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv17 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv18 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        #coming up
       
        #last layer leftttt

        self.dropout1 = nn.Dropout3d(0.1)
        self.dropout2 = nn.Dropout3d(0.2)
        self.dropout3 = nn.Dropout3d(0.3)
        #dropout
        
    def forward(self, x):
        
        out1 = self.elu(self.conv2(self.dropout1(self.elu(self.conv1(x))
        out2 = self.maxpool(out1)
        out3 = self.elu(self.conv4(self.dropout1(self.elu(self.conv3(out2))
        out4 = self.maxpool(out3)
        out5 = self.elu(self.conv6(self.dropout2(self.elu(self.conv5(out4))
        out6 = self.maxpool(out5)
        out7 = self.elu(self.conv8(self.dropout2(self.elu(self.conv7(out6))
        out8 = self.maxpool(out7)
        #going down
                                                
        out9 = self.elu(self.conv10(self.dropout3(self.elu(self.conv9(out8))
        #lowermost block
                                               
        out10 = self.convT1(out9)
        out11 = torch.cat(out10,out7) 
        out12 = self.elu(self.conv12(self.dropout2(self.elu(self.conv11(out11))                                
        out13 = self.convT2(out12)
        out14 = torch.cat(out13,out5)
        out15 = self.elu(self.conv14(self.dropout2(self.elu(self.conv13(out14))
        out16 = self.convT3(out15)
        out17 = torch.cat(out16,out3)
        out18 = self.elu(self.conv16(self.dropout1(self.elu(self.conv15(out17))
        out19 = self.convT4(out18)
        out20 = torch.cat(out19,out1)
        out21 = self.elu(self.conv18(self.dropout1(self.elu(self.conv17(out20))
                                                  
        out22 = #8 channels to 1 channel
        
        return out22
