# file used to generate para files to run arrays of DENSE jobs 
# following the template : 
# RUN_NAME ENV lr model dataset beta seed num_users l2_norm_clip noise_multiplier minibatch_size microbatch_size LDP iterations local_ep delta batch_size synthesis_batch_size lr_g bn oh T save_dir other adv epochs g_steps iid lr_dist beta_dist seed_dist

RUN_NAME = input("RUN_NAME")
PYHTON_FILE = input("PYTHON_FILE")
ENV = input("ENV")
lr = input("lr")
model = input("model")
dataset = input("dataset")
beta = input("beta")
seed = input("seed")
num_users = input("num_users")
l2_norm_clip = input("l2_norm_clip")
noise_multiplier = input("noise_multiplier")
minibatch_size = input("minibatch_size")
microbatch_size = input("microbatch_size")
LDP = input("LDP")
iterations = input("iterations")
local_ep = input("local_ep")
delta = input("delta")
batch_size = input("batch_size")
synthesis_batch_size = input("synthesis_batch_size")
lr_g = input("lr_g")
bn = input("bn")
oh = input("oh")
T = input("T")
save_dir = input("save_dir")
other = input("other")
adv = input("adv")
epochs = input("epochs")
g_steps = input("g_steps")
iid = input("iid")
lr_dist = input("lr_dist")
beta_dist = input("beta_dist")
seed_dist = input("seed_dist")

param_line = RUN_NAME + " " + PYHTON_FILE + " " + ENV + " " + lr + " " + model + " " + dataset + " " + beta + " " + seed + " " + num_users + " " + l2_norm_clip + " " + noise_multiplier + " " + minibatch_size + " " + microbatch_size + " " + LDP + " " + iterations + " " + local_ep + " " + delta + " " + batch_size + " " + synthesis_batch_size + " " + lr_g + " " + bn + " " + oh + " " + T + " " + save_dir + " " + other + " " + adv + " " + epochs + " " + g_steps + " " + iid + " " + lr_dist + " " + beta_dist + " " + seed_dist

print(param_line)









