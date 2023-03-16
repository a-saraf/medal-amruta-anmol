class unet(nn.Module):
    
    def __init__(self, ):
        super(unet, self).__init__()
        
        self.maxpool = nn.MaxPool3d((2, 2, 2))
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        
        self.linear1 = nn.Linear(4800, 2000)
        self.dropout1 = nn.Dropout3d(0.5)
        
        self.linear2 = nn.Linear(2000, 500)
        self.dropout2 = nn.Dropout3d(0.5)
        
        self.linear3 = nn.Linear(500, 3)
        
    def forward(self, x):
        
        out = self.maxpool(self.conv1(x))
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
inputs = Input((240,240,155,1))


c1 = Conv3D(1, 16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv3D(16, 16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv3D(16, 32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv3D(32, 32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv3D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.1) (c3)
c3 = Conv3D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv3D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv3D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.2) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

#The lowermost layer has two convolutional layers and a dropout layer
c6 = Conv3D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p5)
c6 = Dropout(0.3) (c6)
c6 = Conv3D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

#The upscaling layers have a convolutional layer, then the previous same size output is concatenated(a skip connection),
#then another convolutional layer, a dropout layer and another convolutional layer
#Five such blocks
u7 = Conv3DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c5])
c7 = Conv3D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv3D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv3DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c4])
c8 = Conv3D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.2) (c8)
c8 = Conv3D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv3DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c3])
c9 = Conv3D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv3D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

u10 = Conv3DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c9)
u10 = concatenate([u10, c2])
c10 = Conv3D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u10)
c10 = Dropout(0.1) (c10)
c10 = Conv3D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c10)

u11 = Conv3DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c10)
u11 = concatenate([u11, c1], axis=3)
c11 = Conv3D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u11)
c11 = Dropout(0.1) (c11)
c11 = Conv3D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c11)

#Using the sigmoid activation function in the final layer
outputs = Conv3D(1, (1, 1), activation='sigmoid') (c11)

#Compiling the model with adam optimizer, learning rate 0.0003, bce dice loss as the loss and dice loss as the metric
model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(lr=0.0003), loss=bce_dice_loss, metrics=[dice_loss])
model.summary()
# Reference: https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4
