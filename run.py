import utils
from torch.optim import Adam

if utils.args.method == "source":
  model = utils.get_model(load_saved_model=True).eval()

elif utils.args.method == "bn":
  model = utils.get_model(load_saved_model=True).train()

elif utils.args.method == "tent":
  import tent
  
  model = utils.get_model(load_saved_model=True, find_unused_parameters=True).train()
  model = tent.configure_model(model)
  tent.check_model(model)

  params, _ = tent.collect_params(model)
  optimizer = Adam(params, lr=utils.args.lr)
  model = tent.Tent(model, optimizer)

elif utils.args.method == "cotta":
  import cotta

  model = utils.get_model(load_saved_model=True)
  model = cotta.configure_model(model)
  cotta.check_model(model)

  params, _ = cotta.collect_params(model)
  optimizer = Adam(params, lr=utils.args.lr)
  model = cotta.CoTTA(model, optimizer, utils.device, mt_alpha=utils.args.mt_alpha, rst_m=utils.args.rst_m)
  
utils.eval(model)