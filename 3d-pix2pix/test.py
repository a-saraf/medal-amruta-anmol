import torch
import time
from dataset import create_dataset
from unet3d import unet
from gan3d import Discriminator
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#LOAD MODEL
gen_model = unet()
path = '../ckpt_models/gen_models/gen_model_last.pth'
gen_model.load_state_dict(torch.load(path))
dis_model= Discriminator()
path = '../ckpt_models/dis_models/dis_model_last.pth'
dis_model.load_state_dict(torch.load(path))


#TEST DATASET
test_preop_dir = '../DATA/00_Test/'
test_postop_dir = '../DATA/01_Test/'
test_dir = {"pre":test_preop_dir, "post":test_postop_dir}
test_dataset = create_dataset(test_dir)
test_dataset_size = len(test_dataset)

generator_loss = torch.nn.BCELoss()
discriminator_loss = torch.nn.BCELoss()

if device == "cuda":
    gen_model.to(device)
    dis_model.to(device)
    generator_loss.to(device)
    discriminator_loss.to(device)

optimizer_G = torch.optim.Adam(gen_model.parameters())
optimizers_D = torch.optim.Adam(dis_model.parameters())
epochs = 500
Tensor = torch.cuda.FloatTensor if (device == "cuda") else torch.FloatTensor
last_loss = 1000
patience = 5
triggertimes = 0

# Testing code
loss_G_total = 0
loss_D_total = 0
    
with torch.no_grad():
    for i, data in enumerate(test_dataset):
        pre_data = data["pre"]
        post_data = data["post"]

        valid = Variable(Tensor(1, 2025).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(1, 2025).fill_(0.0), requires_grad=False)

        pre_data = Variable(pre_data.type(Tensor))
        post_data = Variable(post_data.type(Tensor))
        generated_postop_data = gen_model(pre_data)
        
        concat_data = torch.cat([pre_data, generated_postop_data], axis=1)

        loss_G = generator_loss(dis_model(concat_data), valid)
            
        real_loss = discriminator_loss(dis_model(torch.cat([pre_data, post_data], axis=1)), valid)
        fake_loss = discriminator_loss(dis_model(concat_data.detach()), fake)
        loss_D = (real_loss + fake_loss) / 2

        loss_G_total += loss_G.item()
        loss_D_total += loss_D.item()

loss_G_total /= test_dataset_size
loss_D_total /= test_dataset_size

print( "loss_G", loss_G.item(), "loss_D", loss_D.item())

file_log = open("test_log.txt","a")
file_log.write(" loss_G "+ str(loss_G.item()) + " loss_D " + str(loss_D.item()) + '\n')
file_log.close()

print('------------------------------------------------------------------------------------------------')
