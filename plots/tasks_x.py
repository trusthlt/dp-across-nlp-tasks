import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "summary_results.csv"
dataframe = pd.read_csv(path, skiprows=0)

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2)
axes = [ax1, ax2, ax3, ax4]
col_names = ['e1', 'e2', 'e5', 'none']
colors = ['#e66747', '#4781e6', '#6ce647', '#e647a6']
labels = ["Ɛ=1", "Ɛ=2", "Ɛ=5", "Ɛ=∞"]
tasks = ['SA', 'NLI', 'NER (CoNLL)', 'NER (Wikiann)', 'POS (GUM)', 'POS (EWT)', 'QA']
models = ['Tr, none, LSTM', 'Tr, none', 'Tr, last two', 'Tr, all'] 

# create the position of each bar
dist = 0.2
num_tasks = len(tasks)
num_models = len(models)
X = []
cur_pos = 1
for j in range(num_tasks):
    for i in range(num_models):
        cur_pos = round(cur_pos + 0.2, 2)
        X.append(cur_pos)
    cur_pos += 0.8

# iterate over each epsilon to fill each plot
for i, (ax, col) in enumerate(zip(axes, col_names)):
    ax.bar(X, dataframe[col], 0.1, color=colors)
    ax.set_title(labels[i])
    ax.set_ylabel("F1")
    ax.set_xlabel("Tasks")
    # create legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(colors))]
    ax.legend(handles, models,ncol=4)

# set the x axis to name the tasks
avgs = []
start = 0
end = 5

for model_n in range(num_tasks):
    avgs.append(np.median(X[start:end+1]))
    start += num_models + 1 
    end += num_models + 1

for ax in axes:  
    plt.setp(ax, xticks=avgs, xticklabels=tasks)



plt.show()
