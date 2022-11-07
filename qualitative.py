# %%
import utils
import core50
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import torch

class_names = ["plug_adapter1","plug_adapter2","plug_adapter3","plug_adapter4","plug_adapter5",
"mobile_phone1","mobile_phone2","mobile_phone3","mobile_phone4","mobile_phone5",
"scissor1","scissor2","scissor3","scissor4","scissor5",
"light_bulb1","light_bulb2", "light_bulb3","light_bulb4","light_bulb5",
"can1","can2" ,"can3","can4","can5", "glass1", "glass2", "glass3", "glass4", "glass5",
"ball1", "ball2", "ball3", "ball4", "ball5",
"marker1", "marker2", "marker3", "marker4", "marker5",
"cup1", "cup2", "cup3", "cup4", "cup5",
"remote_control1", "remote_control2", "remote_control3", "remote_control4", "remote_control5"]


utils.args.path = "/dataQ/tommie_kerssies"
utils.args.seed = 2020
utils.args.model = "best_1_2020.pth.tar"
utils.args.sources = [1]
utils.args.targets = [10]
utils.args.batch_size = 400
utils.dataset_ = core50
model = utils.get_model(load_saved_model=True)
loader = utils.get_loader(domains=utils.args.targets, include_train_data=True, include_val_data=True)
model.train()
# plot model predictions
for x, y in loader:
  x = x.to(utils.device)
  y_pred = model(x).detach().cpu()
  hard_pred = torch.max(y_pred, dim=1).indices
  y_pred = softmax(y_pred, dim=1)
  y_pred = y_pred.numpy()
  for i in range(5, 10):
    plt.imshow(x[i].detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()
    plt.close()
    print("ground truth: " + str(y[i].numpy()) + class_names[int(y[i])])
    print("predicted class: " + str(hard_pred[i].numpy()) + class_names[int(hard_pred[i])])
    plt.bar(range(50), y_pred[i])
    plt.show()
  break