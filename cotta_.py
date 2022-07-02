# %%
import utils
import torch
import cotta

model = utils.get_model(load_saved_model=True)
model = cotta.configure_model(model)
params, _ = cotta.collect_params(model)
optimizer = torch.optim.Adam(params, lr=utils.args.lr)
model = cotta.CoTTA(model, optimizer, utils.device, mt_alpha=.9, rst_m=0.1, ap=0.)
utils.eval(model, eval_mode=False, reset=True, stop_permutation=1)