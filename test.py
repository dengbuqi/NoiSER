import torch
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from model import NoiSER

model = NoiSER().cuda()
model.load_state_dict(torch.load('./last.pt'))
model.eval()

def test(img_path):
    img_name = img_path.split('/')[-1]
    print(img_name)
    with torch.no_grad():
        I = read_image(img_path).cuda()/255
        to_pil_image(I).save(f'./{img_name}input.png')
        I = I.unsqueeze(0)
        I = I*2-1
        enhanced = model(I)
        enhanced = (enhanced+1)/2
        to_pil_image(enhanced[0]).save(f'./{img_name}enhanced.png')

print('Start TEST....')
# test('../Math/1_night.png')
test('./low00033.png')
print('End TEST....')
