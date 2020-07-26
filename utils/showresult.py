
# In[1]
import numpy as np
import matplotlib.pyplot as plt
from model.Unet import Unet
from model.SegNet import SegNet
from model.PSPNet import PSPNet
from model.Deeplab import Deeplab
from model.DeconvNet import DeconvNet
from model.ENet import ENet
from dataloader import DataLoader
from utils import COLOR_DICT, show_label, show_predict_image
from matplotlib.pyplot import savefig

unet_model = Unet((512, 512, 3), 12)
weight_path = 'Log/Unet/weight.h5'
unet_model.load_weights(weight_path)

segnet_model = SegNet((512, 512, 3), 12)
weight_path = 'Log/SegNet/weight.h5'
segnet_model.load_weights(weight_path)

pspnet_model = PSPNet((480, 480, 3), 12)
weight_path = 'Log/PSPNet/weight.h5'
pspnet_model.load_weights(weight_path)

deeplab_model = Deeplab((512, 512, 3), 12)
weight_path = 'Log/Deeplab/weight.h5'
deeplab_model.load_weights(weight_path)

deconvnet_model = DeconvNet((512, 512, 3), 12)
weight_path = 'Log/DeconvNet/weight.h5'
deconvnet_model.load_weights(weight_path)

enet_model = ENet((512, 512, 3), 12)
weight_path = 'Log/ENet/weight.h5'
enet_model.load_weights(weight_path)


# In[2]

train_path = 'G:\备份\备份\dataset\CamVid'
val_path = 'G:\备份\备份\dataset\CamVid'
train_img_folder = 'train'
train_label_folder = 'trainannot'
val_image_folder = 'val'
val_label_folder = 'valannot'
image_size_ = (480, 480)
label_size_ = (240, 240)
image_size = (512, 512)
label_size = (512, 512)
color_mode = 'rgb'
num_classes = 12

dataloader1 = DataLoader(train_path, val_path, train_img_folder,
                         train_label_folder, val_image_folder, val_label_folder,
                         num_classes, image_size, label_size, batch_size=5)
dataloader2 = DataLoader(train_path, val_path, train_img_folder,
                         train_label_folder, val_image_folder, val_label_folder,
                         num_classes, image_size_, label_size_, batch_size=5)

data1 = dataloader1.validData()
data2 = dataloader2.validData()
x, y = next(data1)
x_, y_ = next(data2)
x.shape


# In[3]
unet_result = unet_model.predict(x, verbose=1, batch_size=1)
segnet_result = segnet_model.predict(x, verbose=1, batch_size=1)
deeplab_reslt = deeplab_model.predict(x, verbose=1, batch_size=1)
deconvnet_result = deconvnet_model.predict(x, verbose=1, batch_size=1)
pspnet_result = pspnet_model.predict(x_, verbose=1, batch_size=1)
enet_result = enet_model.predict(x, verbose=1, batch_size=1)
enet_result.shape

# In[5]
enet_result.resize(5, 512, 512, 12)
enet_result.shape
# In[4]
fig3 = plt.figure(constrained_layout=True)
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300
gs = fig3.add_gridspec(5, 8)


for i in range(5):
    f = fig3.add_subplot(gs[i, 0])
    # if i == 0:
    # f.set_title("pic")
    plt.imshow(x[i])
    plt.axis('off')

for i in range(5):
    f = fig3.add_subplot(gs[i, 1])
    # if i == 0:
    # f.set_title("label")
    show_label(y[i], (512, 512))
    plt.axis('off')

for i in range(5):
    f = fig3.add_subplot(gs[i, 2])
    # if i == 0:
    # f.set_title("unet")
    show_predict_image(unet_result[i], (512, 512))
    plt.axis('off')

for i in range(5):
    f = fig3.add_subplot(gs[i, 3])
    # if i == 0:
    # f.set_title("segnet")
    show_predict_image(segnet_result[i], (512, 512))
    plt.axis('off')

for i in range(5):
    f = fig3.add_subplot(gs[i, 4])
    # if i == 0:
    # f.set_title("pspnet")
    show_predict_image(pspnet_result[i], (240, 240))
    plt.axis('off')

for i in range(5):
    f = fig3.add_subplot(gs[i, 5])
    # if i == 0:
    # f.set_title("deeplab")
    show_predict_image(deeplab_reslt[i], (512, 512))
    plt.axis('off')

for i in range(5):
    f = fig3.add_subplot(gs[i, 6])
    # if i == 0:
    # f.set_title("deconvnet")
    show_predict_image(deconvnet_result[i], (512, 512))
    plt.axis('off')

for i in range(5):
    f = fig3.add_subplot(gs[i, 7])
    # if i == 0:
    # f.set_title("enet")
    show_predict_image(enet_result[i], (512, 512))
    plt.axis('off')

savefig('pic.png')
plt.show()


# %%
show_predict_image(deeplab_reslt[0], (512, 512))
plt.show()

show_predict_image(segnet_result[0], (512, 512))
plt.show()
# %%
show_label(y[0], (512, 512))
plt.show()
