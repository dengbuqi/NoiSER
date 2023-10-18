import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from dataloader import GaussianImageDataset
from loss import L_TV, L_LMCV, L_DARK, L_exp
from model import NoiSER
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

LENGTH = 2000
SNAPSHOT_ITER = 100
IMG_SIZE = (104, 104)
is_log = True

data_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        # transforms.ToTensor(),
    ])
custom_dataset = GaussianImageDataset(length=LENGTH, img_size=IMG_SIZE, mean=0, stddev=3, transform=data_transform)
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=True, num_workers=4)
log_path = './log/'

l_tv = L_TV()
l1loss = nn.L1Loss()
# llmcv = L_LMCV(1)
# ldark = L_DARK(0.5)
# lexp = L_exp()

model = NoiSER().cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))

tbar = tqdm(dataloader)
i = 0
l1s = []
ltvs = []
losses = []
# llmcvs = []
# ldarks = []
# lexps = []


for images in tbar:
    # to_pil_image(images[0]).save(f'{log_path}{i}.png')
    tbar.set_description_str(f'NiSER Training: {i+1}/{LENGTH} iteration')
    images = images.cuda()
    enhanced = model(images)
    loss_tv = l_tv(enhanced)
    loss_l1 = l1loss(enhanced,images)
    # loss_lmcv = 0.5*llmcv((enhanced+1)/2) # range to [0,1]
    # loss_dark = ldark((enhanced+1)/2) # range to [0,1]
    # loss_exp = 10*lexp((enhanced+1)/2, 0.6)
    loss = loss_l1 + loss_tv #+ loss_exp + loss_lmcv + loss_dark 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_l1_value = loss_l1.item()
    loss_tv_value = loss_tv.item()
    # loss_lmcv_value = loss_lmcv.item()
    # loss_dark_value = loss_dark.item()
    # loss_exp_value = loss_exp.item()

    loss_value = loss.item()
    l1s.append(loss_l1_value)
    ltvs.append(loss_tv_value)
    # llmcvs.append(loss_lmcv_value)
    # ldarks.append(loss_dark_value)
    # lexps.append(loss_exp_value)

    losses.append(loss_value)

    tbar.set_postfix_str(f'loss={loss_value:0.4f}'
                        +f'|L1={loss_l1_value:0.4f}'
                        +f'|TV={loss_tv_value:0.4f}'
                        # +f'|LMCV={loss_lmcv_value:0.4f}'
                        # +f'|DARK={loss_dark_value:0.4f}'
                        # +f'|EXP={loss_exp_value:0.4f}'
                        )

    if ((i+1) % SNAPSHOT_ITER) == 0:
        torch.save(model.state_dict(), './last.pt')
    i+=1

if is_log:
    data = {'losses':losses,
            'l1s':l1s,
            'ltvs':ltvs, 
            # 'llmcv':llmcvs,
            # 'ldark':ldarks,
            # 'lexp':lexps,
            }
    df = pd.DataFrame(data)
    df.to_csv("log.csv", index=False)
    plt.plot(l1s, label='l1s', alpha=0.5)
    plt.plot(ltvs, label='ltvs', alpha=0.5)
    plt.plot(losses, label='looses', alpha=0.5)
    # plt.plot(llmcvs, label='llmcv', alpha=0.5)
    # plt.plot(ldarks, label='ldark', alpha=0.5)
    # plt.plot(lexps, label='lexp', alpha=0.5)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.savefig('./log.png')