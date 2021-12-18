from opacus.privacy_analysis import compute_rdp, get_privacy_spent
import numpy as np

Tr_train_size_NLI = 549360
#Tr_train_set_size_QA = 102123
train_size = Tr_train_size_NLI
batch_size = 32
sample_rate = batch_size/train_size
noise_multiplier = 0.7350
epochs = 30
steps = epochs * (train_size/batch_size)
orders = [10]
delta = 1e-5
for order in orders:
    rdp = compute_rdp(sample_rate, noise_multiplier, steps, float(order))
    epsilon, opt_order = get_privacy_spent(float(order), rdp, delta)
    print("order: ", order, "ε: ", epsilon, 'α',opt_order)
