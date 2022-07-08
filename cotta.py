# %%
from copy import deepcopy
import torch
import torch.jit
from torch.nn import BatchNorm2d, SyncBatchNorm, Module
import torchvision.transforms as transforms
import my_transforms as my_transforms
import utils
import torch
from domainnet import img_size
import matplotlib.pyplot as plt


class CoTTA(Module):
  """CoTTA adapts a model by entropy minimization during testing.

  Once tented, a model adapts itself by updating on every forward.
  """

  def __init__(self, model, optimizer, device, mt_alpha, rst_m, ap=None, steps=1, episodic=False):
    super().__init__()
    self.model = model
    self.optimizer = optimizer
    self.steps = steps
    assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
    self.episodic = episodic

    self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
        copy_model_and_optimizer(self.model, self.optimizer)
    self.transform = get_tta_transforms()
    self.mt = mt_alpha
    self.rst = rst_m
    self.ap = ap
    self.device = device

  def forward(self, x):
    if self.episodic:
      self.reset()

    for _ in range(self.steps):
      outputs = self.forward_and_adapt(x)

    return outputs

  def reset(self):
    if self.model_state is None or self.optimizer_state is None:
      raise Exception("cannot reset without saved model/optimizer state")
    load_model_and_optimizer(self.model, self.optimizer,
                             self.model_state, self.optimizer_state)
    # Use this line to also restore the teacher model
    self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
        copy_model_and_optimizer(self.model, self.optimizer)

  @torch.enable_grad()  # ensure grads in possible no grad context for testing
  def forward_and_adapt(self, x):
    outputs = self.model(x)
    # Teacher Prediction
    # anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
    standard_ema = self.model_ema(x)
    # Threshold choice discussed in supplementary
    # if anchor_prob.mean(0) < self.ap:
    #   # Augmentation-averaged Prediction
    #   N = 32
    #   outputs_emas = []
    #   for _ in range(N):
    #     # plt.imshow(x[2].permute(1, 2, 0).cpu().numpy())
    #     # plt.show()
    #     # plt.imshow(self.transform(x)[2].permute(1, 2, 0).cpu().numpy())
    #     # raise Exception()
    #     outputs_ = self.model_ema(self.transform(x)).detach()
    #     outputs_emas.append(outputs_)
    #   outputs_ema = torch.stack(outputs_emas).mean(0)
    # else:
    #   outputs_ema = standard_ema
    outputs_ema = standard_ema
    # Student update
    loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    # Teacher update
    self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
    # Stochastic restore
    for nm, m in self.model.named_modules():
      for npp, p in m.named_parameters():
        if npp in ['weight', 'bias'] and p.requires_grad:
          mask = (torch.rand(p.shape) < self.rst).float().to(self.device)
          with torch.no_grad():
            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
    return outputs_ema


def get_tta_transforms(gaussian_std: float = 0.005, soft=False):
  tta_transforms = transforms.Compose([
      my_transforms.Clip(0.0, 1.0),
      my_transforms.ColorJitterPro(
          brightness=[0.8, 1.2] if soft else [0.6, 1.4],
          contrast=[0.85, 1.15] if soft else [0.7, 1.3],
          saturation=[0.75, 1.25] if soft else [0.5, 1.5],
          hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
          gamma=[0.85, 1.15] if soft else [0.7, 1.3]
      ),
      transforms.Pad(padding=int(img_size / 2), padding_mode='edge'),
      transforms.RandomAffine(
          degrees=[-8, 8] if soft else [-15, 15],
          translate=(1/16, 1/16),
          scale=(0.95, 1.05) if soft else (0.9, 1.1),
          shear=None,
          interpolation=transforms.InterpolationMode.BILINEAR,
          fillcolor=None
      ),
      transforms.GaussianBlur(kernel_size=5, sigma=[
                              0.001, 0.25] if soft else [0.001, 0.5]),
      transforms.CenterCrop(size=img_size),
      transforms.RandomHorizontalFlip(p=0.5),
      my_transforms.GaussianNoise(0, gaussian_std),
  ])
  return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
  for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + \
        (1 - alpha_teacher) * param[:].data[:]
  return ema_model


@torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
  """Entropy of softmax distribution from logits."""
  return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
  """Collect all trainable parameters.

  Walk the model's modules and collect all parameters.
  Return the parameters and their names.

  Note: other choices of parameterization are possible!
  """
  params = []
  names = []
  for nm, m in model.named_modules():
    for np, p in m.named_parameters():
      if np in ['weight', 'bias'] and p.requires_grad:
        params.append(p)
        names.append(f"{nm}.{np}")
  return params, names


def copy_model_and_optimizer(model, optimizer):
  """Copy the model and optimizer states for resetting after adaptation."""
  # deepcopy on weightnorm gives an error, so fix it by detaching before and reattaching after
  module = model.module if utils.distributed else model
  weight_norm_used = 'fc' in [x[0] for x in module.named_children()] and hasattr(
      module.fc, 'weight_g') and hasattr(module.fc, 'weight_v')
  if weight_norm_used:
    module.fc.weight = module.fc.weight_v.detach()

  model_state = deepcopy(model.state_dict())
  optimizer_state = deepcopy(optimizer.state_dict())
  model_anchor = deepcopy(model)
  ema_model = deepcopy(model)

  for param in ema_model.parameters():
    param.detach_()

  return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
  """Restore the model and optimizer states from copies."""
  model.load_state_dict(model_state, strict=True)
  optimizer.load_state_dict(optimizer_state)


def configure_model(model):
  """Configure model for use with tent."""
  # train mode, because tent optimizes the model to minimize entropy
  model.train()
  # disable grad, to (re-)enable only what we update
  model.requires_grad_(False)
  # enable all trainable
  for m in model.modules():
    m.requires_grad_(True)
    if isinstance(m, (BatchNorm2d, SyncBatchNorm)):
      # force use of batch stats in train and eval modes
      m.track_running_stats = False
      m.running_mean = None
      m.running_var = None
  return model


def check_model(model):
  """Check model for compatability with tent."""
  is_training = model.training
  assert is_training, "tent needs train mode: call model.train()"
  param_grads = [p.requires_grad for p in model.parameters()]
  has_any_params = any(param_grads)
  has_all_params = all(param_grads)
  assert has_any_params, "tent needs params to update: " \
                         "check which require grad"
  assert has_all_params, "cotta should update all params"