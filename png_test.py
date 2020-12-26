from PIL import Image
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms import transforms

transform = transforms.Compose([
    # transforms.Scale(size),
    transforms.Resize((224, 224)),
    # transforms.CenterCrop((299, 299)),
    # transforms.RandomRotation(0.1),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # PIL Image → Tensor
])


path = r'C:\Users\99785\Desktop\my.jpg'
img = Image.open(path)
plt.imshow(img)
plt.show()
img = img.convert("BRG")
plt.imshow(img)
plt.show()
# img = np.array(img)
img = transform(img)
# img = io.imread(image_url)  # ndarray
# img = Image.fromarray(img)  # 转化为PIL格式
print(type(img), img.shape[0])