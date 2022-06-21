# %%
import utils
import torch
from tqdm import tqdm
import cotta

model = utils.get_model()
model = cotta.configure_model(model)
params, _ = cotta.collect_params(model)
optimizer = torch.optim.Adam(params, lr=utils.args.lr)
model = cotta.CoTTA(model, optimizer, utils.device, mt_alpha=1., rst_m=1., ap=1.)
utils.eval(model, reset=True, stop_permutation=1)