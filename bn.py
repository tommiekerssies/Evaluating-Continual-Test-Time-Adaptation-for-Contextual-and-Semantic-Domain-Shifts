# %%
import utils

model = utils.get_model(load_saved_model=True) 
utils.eval(model, eval_mode=False) # use training mode for BN, so the model will use batch statistics