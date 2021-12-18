import os
import sys
"""
acts as a wrapper function to call the actual programm given the input and output
"""
def main(input_, output_):
    seed = 8672
    lr = 0.00001
    dp = 1
    BLay = 0
    rnn = 0
    test_out = output_
    test_in = input_
    out_ = "<output dir>"
    BS = 4
    eps = "NONE"
    delta = 1e-05
    nm = 0.7186
    alpha = 10
    ES = 0
    model_path = "<path to pytorch model (pt file)>"
    stream = os.popen(f'python <pathToRepo>/dp-across-nlp-tasks/NLPCode/question_answering/test_suit/experiment_parameters.py --seed {seed} --lr {lr} --dp {dp} \
    --BLay {BLay} --rnn {rnn}  --out {out_} --dsDir {test_in} --BS {BS} --eps {eps}  --del {delta}  --noiseM {nm} --alpha {alpha} --ES {ES} \
    --test-out {test_out} --test-in {test_in} --modelP {model_path}')
    print("execute EP")
    output = stream.readlines()

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1])