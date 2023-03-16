class unet(nn.Module):
    
    def __init__(self, ):
        super(unet, self).__init__()
        
        self.elu=nn.ELU()
        self.maxpool = nn.MaxPool3d((2, 2, 2))
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv7 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)
        
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv1 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3)
        
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        
        self.dropout1 = nn.Dropout3d(0.1)
        self.dropout2 = nn.Dropout3d(0.2)
        self.dropout3 = nn.Dropout3d(0.3)
        
    def forward(self, x):
        
        out = self.maxpool(self.elu(self.conv2(self.dropout1(self.elu(self.conv1(x)))
        out = self.maxpool(self.elu(self.conv4(self.dropout1(self.elu(self.conv3(out)))
        out = self.maxpool(self.elu(self.conv6(self.dropout2(self.elu(self.conv5(out)))
        out = self.maxpool(self.elu(self.conv8(self.dropout2(self.elu(self.conv7(out)))
                                               
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.maxpool(self.conv2(out))
        out = self.maxpool(self.conv3(out))
        
        # Flattening process
        b, c, d, h, w = out.size() # batch_size, channels, depth, height, width
        out = out.view(-1, c * d * h * w)
        
        out = self.dropout1(self.linear1(out))
        out = self.dropout2(self.linear2(out))
        out = self.linear3(out)
        
        out = torch.softmax(out, 1)
        
        return out




#Input image shape provided
inputs = Input((256,256,3))

#Wraps arbitrary expressions as a Layer object.
s = Lambda(lambda x: x / 255) (inputs)

#The downscaling layers have one convolutional layer, then a dropout layer, 
#then another convolutional layer and the maxpooling to reduce the size of the image
#Four such blocks
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

#The lowermost layer has two convolutional layers and a dropout layer
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

#The upscaling layers have a convolutional layer, then the previous same size output is concatenated(a skip connection),
#then another convolutional layer, a dropout layer and another convolutional layer
#Four such blocks
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

#Using the sigmoid activation function in the final layer
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

#Compiling the model with adam optimizer, learning rate 0.0003, bce dice loss as the loss and dice loss as the metric
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(lr=0.0003), loss=bce_dice_loss, metrics=[dice_loss])
model.summary()
# Reference: https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4
