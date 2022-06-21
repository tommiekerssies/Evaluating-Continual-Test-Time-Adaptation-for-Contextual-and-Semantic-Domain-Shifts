# %%
import utils

model = utils.get_model().train() # use training mode for BN, so the model will use batch statistics
utils.eval(model, reset=False, stop_permutation=1)