# %%
import utils

model = utils.get_model().eval()
utils.eval(model, cycles=1, stop_permutation=1)