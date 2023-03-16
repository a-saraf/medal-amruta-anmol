import torch
import time
from dataset import create_dataset
from unet3d import unet
from gan3d import Discriminator
from torch.autograd import Variable

preop_dir = '../DATA/00_Train/'
postop_dir = '../DATA/01_Train/'

dir = {"pre":preop_dir, "post":postop_dir}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = create_dataset(dir)
dataset_size = len(dataset)

gen_model = unet()
dis_model = Discriminator()

generator_loss = torch.nn.BCELoss()
discriminator_loss = torch.nn.BCELoss()

if device == "cuda":
    gen_model = torch.nn.DataParallel(gen_model)
    dis_model = torch.nn.DataParallel(dis_model)
    gen_model.to(device)
    dis_model.to(device)
    generator_loss.to(device)
    discriminator_loss.to(device)

optimizer_G = torch.optim.Adam(gen_model.parameters())
optimizers_D = torch.optim.Adam(dis_model.parameters())

epochs = 500

Tensor = torch.cuda.FloatTensor if (device == "cuda") else torch.FloatTensor

for epoch in range(epochs):
    start_time = time.time()
    for i, data in enumerate(dataset):
        
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

    print("epoch", epoch, "loss_G", loss_G.item(), "loss_D", loss_D.item())

    file_log = open("train_log.txt","a")
    file_log.write("epoch," + str(epoch) + "loss_G,"+ str(loss_G.item()) + "loss_D," + str(loss_D.item()) + '\n')
    file_log.close()

    path = '../ckpt_models/gen_models/gen_model_last.pth'
    torch.save(gen_model, path)
    path = '../ckpt_models/dis_models/dis_model_last.pth'
    torch.save(dis_model, path)

    print(time.time() - start_time)

