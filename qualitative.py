# %%
import utils
import core50
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import torch
from torchvision import transforms
from torch.optim import Adam

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

objects = ['cup4', 'light_bulb5', 'can2', 'scissor5', 'remote_control3', 
           'glass4', 'mobile_phone5', 'cup2', 'remote_control5', 'can4'] 

inv_norm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
          transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                std = [ 1., 1., 1. ]),
          ])
# %%
obj_class_temp = []
for object in objects:
  # get index of object in class_names
  obj_class_temp.append(class_names.index(object))  
utils.args.batch_size = 1
source_loader = utils.get_loader(domains=utils.args.sources, include_train_data=False, include_val_data=True)
utils.args.batch_size = 400
source_examples_x = []
source_examples_y = []
for i, (x, y) in enumerate(source_loader):
  if len(obj_class_temp) == 0:
    break
  if y[0].cpu().numpy() in obj_class_temp:
    source_examples_x.append(x[0])
    source_examples_y.append(y[0])
    obj_class_temp.remove(y[0].cpu().numpy())
source_examples_x = torch.stack(source_examples_x).to(utils.device)
source_examples_y = torch.stack(source_examples_y)
# %%
model = utils.get_model(load_saved_model=True).train()
# %%
import tent
model = utils.get_model(load_saved_model=True).train()
model = tent.configure_model(model)
tent.check_model(model)
params = tent.Tent.collect_params(model)
optimizer = Adam(params, lr=0.001)
model = tent.Tent(model, optimizer)
# %%
import cotta
model = utils.get_model(load_saved_model=True).train()
model = cotta.configure_model(model)
cotta.check_model(model)
params = cotta.CoTTA.collect_params(model)
optimizer = Adam(params, lr=0.001)
model = cotta.CoTTA(model, optimizer, utils.device, mt_alpha=0.9, rst_m=0)
# %%
# plot source predictions
y_pred = model(source_examples_x).detach().cpu()
hard_pred = torch.max(y_pred, dim=1).indices
y_pred = softmax(y_pred, dim=1)
y_pred = y_pred.numpy()
source_examples_x = source_examples_x.detach().cpu()
source_examples_x = inv_norm(source_examples_x)
correct = 0
for i in range(10):
  if hard_pred[i].numpy() == source_examples_y[i].numpy():
    correct += 1
  plt.imshow(source_examples_x[i].numpy().transpose(1,2,0))
  plt.show()
  plt.close()
  fig1, ax = plt.subplots()
  ax.set_box_aspect(1)
  plt.ylim(0.0, 1.0)
  color = 50 * ['red']
  color[int(source_examples_y[i].numpy())] = 'green'
  plt.bar(range(50), y_pred[i], color=color)
  plt.show()
  print("ground truth: " + str(source_examples_y[i].numpy()) + class_names[int(source_examples_y[i])])
  print("predicted class: " + str(hard_pred[i].numpy()) + class_names[int(hard_pred[i])])
print(correct)
#%%
# plot target predictions
target_loader = utils.get_loader(domains=utils.args.targets, include_train_data=True, include_val_data=True)
for x, y in target_loader:
  x = x.to(utils.device)
  y_pred = model(x).detach().cpu()

hard_pred = torch.max(y_pred, dim=1).indices
y_pred = softmax(y_pred, dim=1)
y_pred = y_pred.numpy()
x = x.detach().cpu()
x = inv_norm(x)
correct = 0
for i in range(30, 40):
  if hard_pred[i].numpy() == y[i].numpy():
    correct += 1
  plt.imshow(x[i].numpy().transpose(1,2,0))
  plt.show()
  plt.close()
  fig1, ax = plt.subplots()
  ax.set_box_aspect(1)
  plt.ylim(0.0, 1.0)
  color = 50 * ['red']
  color[int(y[i].numpy())] = 'green'
  plt.bar(range(50), y_pred[i], color=color)
  plt.show()
  print("ground truth: " + str(y[i].numpy()) + class_names[int(y[i])])
  print("predicted class: " + str(hard_pred[i].numpy()) + class_names[int(hard_pred[i])])
print(correct)
