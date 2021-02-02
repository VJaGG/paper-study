import cv2
import torch
import numpy as np
from fcn import FCN
import torchvision.transforms as transforms


colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0],
            [128, 128, 0], [0, 0, 128], [128, 0, 128],
            [0, 128, 128], [128, 128, 128], [64, 0, 0],
            [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128],
            [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
colormap = np.array(colormap).astype("uint8")
# img_path = "./data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg"
img_path = "./demo/demo.jpg"
path = "./model/best.pth"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
w, h, _ = image.shape
padding0 = 32 - w % 32
padding1 = 32 - h % 32
image = cv2.copyMakeBorder(image,
                           padding0 // 2,
                           padding0 - padding0 // 2,
                           padding1 // 2,
                           padding1 - padding1 // 2,
                           cv2.BORDER_REFLECT)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image = transform(image).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
state_dict = torch.load(path, map_location='cpu')
model = FCN(21).to(device).eval()
model.load_state_dict(state_dict)
del state_dict
with torch.no_grad():
    image = image.to(device)
    output = model(image)

predict = output.data.cpu().squeeze(0).numpy()
predict = np.argmax(predict, axis=0)
predict = colormap[predict]

cv2.imwrite("res.png", predict)
