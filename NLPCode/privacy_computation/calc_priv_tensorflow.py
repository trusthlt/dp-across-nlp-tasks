import os
import sys
import numpy as np
path_to_priv_calc= "<path to library>/lib64/python3.6/site-packages/tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy.py"

train_size_POS_EWT = 12542
train_size_POS_GUM = 5664
train_size_NLI = 549360
train_size_SA = 9686
train_size_NER_wikiaan = 20000 # val, test: 10000

gamma = 0.000001
def equal(x1, x2):
    return abs(x1-x2) <= gamma



def cal_tens(size, bs, nm, e, d):
    call = f'python {path_to_priv_calc} --N={size} --batch_size={bs} --noise_multiplier={nm} --epochs={e} --delta={d}'
    stream = os.popen(call)
    arr = stream.read().split(" ")
    eps = float(arr[20])
    alpha = float(arr[29].replace("The", "").replace(".\n", ""))
    return eps, alpha

def find_best_in_range(target_eps, lower, upper, stepsize, train_size, epochs, delta, batch_size, eps_lower):
    best_eps = 0
    best_diff = float('inf')
    eps_range = [] # array of tuples
    #if not eps_lower == None:
        #eps_range.append((eps_lower, abs(eps_lower-target_eps),lower))
    # try out all, until it is getting worse
    for nm in np.arange(lower, upper + stepsize, stepsize):
        eps, alpha = cal_tens(train_size, batch_size, nm, epochs, delta)
        if equal(eps, target_eps):
            print(f'for eps {eps} use noise multiplier {nm} and alpha {alpha}')
            sys.exit()
        diff = abs(eps-target_eps)
        if diff < best_diff:
            best_eps = eps
            best_diff = diff
            eps_range.append((eps, diff, nm, alpha))
        else:
            eps_range.append((eps, diff, nm, alpha))
            #break
    # find best two fits
    eps_range = sorted(eps_range, key=lambda tup: tup[1])
    # find the best fit that is smaller
    for tup in eps_range:
        if tup[0] > target_eps:
            smaller = tup
            break
    # find the best fit that is larger
    for tup in eps_range:
        if tup[0] < target_eps:
            larger = tup
            break
    return smaller[2], larger[2], smaller[0], larger[0], smaller[3], larger[3]



def main(target_eps):
    train_size = train_size_NER_wikiaan
    epochs = 30
    delta = 1e-5
    batch_size = 42
    nm_lower = 1
    nm_upper = 2

    # check upper -> epsilon larger, nm lower
    eps, alpha = cal_tens(train_size, batch_size, nm_lower, epochs, delta)
    if equal(eps, target_eps):
        print(f'for eps {eps} use noise multiplier {nm_lower} and alpha {alpha}')
        sys.exit()
    if not eps > target_eps:
        print("Bounds are wrong please lower the lower bound.")
        sys.exit()

    # check lower -> eps lower nm higher
    eps, alpha = cal_tens(train_size, batch_size, nm_upper, epochs, delta)
    if equal(eps, target_eps):
        print(f'for eps {eps} use noise multiplier {nm_upper} and alpha {alpha}')
        sys.exit()
    if not eps < target_eps:
        print("Bounds are wrong please make the upper bound larger.")
        sys.exit()
    eps_lower = None
    for stepsize in [0.1, 0.01, 0.001, 0.0001]:
        nm_lower, nm_upper, eps_lower, eps_upper, alpha_lower, alpha_upper = find_best_in_range(target_eps, nm_lower, nm_upper, stepsize, train_size, epochs, delta, batch_size, eps_lower)

    print(f'for eps {target_eps} use noise multiplier between {nm_lower} ({eps_lower}) alpha ({alpha_lower}); and {nm_upper} ({eps_upper}) alpha ({alpha_upper})')



if __name__=="__main__":
    main(1)