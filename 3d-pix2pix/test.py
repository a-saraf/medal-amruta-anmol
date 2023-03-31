import torch
import time
from dataset import create_dataset
from unet3d import unet
from gan3d import Discriminator
from torch.autograd import Variable

test_preop_dir = '../DATA/00_Test/'
test_postop_dir = '../DATA/01_Test/'

test_dir = {"pre":test_preop_dir, "post":test_postop_dir}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dataset = create_dataset(test_dir)
test_dataset_size = len(test_dataset)

gen_model = unet()
dis_model = Discriminator()

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

for epoch in range(epochs):
    gen_model.train()
    dis_model.train()

    start_time = time.time()
    for i, data in enumerate(train_dataset):
        
        pre_data = data["pre"]
        post_data = data["post"]

        valid = Variable(Tensor(1, 2025).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(1, 2025).fill_(0.0), requires_grad=False)

        pre_data = Variable(pre_data.type(Tensor))
        post_data = Variable(post_data.type(Tensor))

        optimizer_G.zero_grad()
        generated_postop_data = gen_model(pre_data)
        
        concat_data = torch.cat([pre_data, generated_postop_data], axis=1)

        loss_G = generator_loss(dis_model(concat_data), valid)
        loss_G.backward()
        optimizer_G.step()

        optimizers_D.zero_grad()

        real_loss = discriminator_loss(dis_model(torch.cat([pre_data, post_data], axis=1)), valid)
        fake_loss = discriminator_loss(dis_model(concat_data.detach()), fake)
        loss_D = (real_loss + fake_loss) / 2

        loss_D.backward()
        optimizers_D.step()

    print("Train:epoch", epoch, "loss_G", loss_G.item(), "loss_D", loss_D.item())

    file_log = open("train_log.txt","a")
    file_log.write("epoch," + str(epoch+1) + "loss_G,"+ str(loss_G.item()) + "loss_D," + str(loss_D.item()) + '\n')
    file_log.close()

    path = '../ckpt_models/gen_models/gen_model_last.pth'
    torch.save(gen_model, path)
    path = '../ckpt_models/dis_models/dis_model_last.pth'
    torch.save(dis_model, path)

    # Validation Part
    loss_G_total = 0
    loss_D_total = 0
    gen_model.eval()
    dis_model.eval()
    
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

    file_log = open("val_log.txt","a")
    file_log.write("epoch " + str(epoch+1) + " loss_G "+ str(loss_G.item()) + " loss_D " + str(loss_D.item()) + '\n')
    file_log.close()

    print('------------------------------------------------------------------------------------------------')
