import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
offset = 3
marker_symb = ["o", "s", "^", "X", "D"]
col_names = ['e1', 'e2', 'e5', 'none', 'std1','std2','std3','stdi']
colors = ['#e66747', '#4781e6', '#6ce647', '#e647a6', '#47e6be', '#97c221', '#21aac2']
x_labels = ["Ɛ=1", "Ɛ=2", "Ɛ=5", "Ɛ=∞"]
tasks = ['SA', 'NLI', 'NER (CoNLL)', 'NER (Wikiann)', 'POS (GUM)', 'POS (EWT)', 'QA']
models = ['Tr, none, LSTM', 'Tr, none', 'Tr, last two', 'Tr, all'] 
num_tasks = len(tasks)
num_models = len(models)

def plot_graph(F1, e_F1, ax, title):
    for task, error, c, m in zip(F1, e_F1, colors, marker_symb):
        ax.errorbar(1, task[0], yerr=error[0], linestyle='None', marker=m, elinewidth=3, color = c)
        ax.annotate(f"{task[0]}", (1,task[0]),  xytext =(offset, -3*offset), textcoords ='offset points')
        ax.errorbar(2, task[1], yerr=error[1], linestyle='None', marker=m, elinewidth=3, color = c)
        ax.annotate(f"{task[1]}", (2,task[1]),  xytext =(offset, -3*offset), textcoords ='offset points')
        ax.errorbar(5, task[2], yerr=error[2], linestyle='None', marker=m, elinewidth=3, color = c)
        ax.annotate(f"{task[2]}", (5,task[2]),  xytext =(offset, -3*offset), textcoords ='offset points')
        ax.errorbar(10, task[3], yerr=error[3], linestyle='None', marker=m, elinewidth=3, color = c)
        ax.annotate(f"{task[3]}", (10,task[3]),  xytext =(offset, -3*offset), textcoords ='offset points')
        ax.plot([1,2,5,10], task, color=c, marker=m)
        ax.set_ylim(0,1)
        ax.set_title(title)
        ax.set_ylabel('F1')
        ax.set_xlabel('Privacy Budged')

# example
'''F1 = [[0.67, 0.68, 0.72], [0.77, 0.78, 0.79],
      [0.66,0.67, 0.69], [0.74, 0.81, 0.84], [0.46, 0.46, 0.46]]
e_F1 = [[0.01, 0.01, 0.01], [0.00, 0.02, 0.02], [0.02, 0.02, 0.03], [0.03, 0.02, 0.02], [0.0, 0.0, 0.03]]
plot_graph(F1, F1_base, e_F1, e_B)
'''

path = "summary_results.csv"
dataframe = pd.read_csv(path, skiprows=0)

# each task one subplot -> 7 plots
fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
axis = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

for i_task, task in enumerate(tasks):
    task_offset = i_task * 4
    F1 = []
    e_F1 = []
    for i in range(num_models):
        F1_temp = []
        e_F1_temp = []
        # add the F1 scores for one model with differend DP configs
        for c in col_names[:4]:
            F1_temp.append(dataframe[c][task_offset+i])

        # add the error bar (std) for one model with differed DP configs
        for c in col_names[4:]:
            e_F1_temp.append(dataframe[c][task_offset+i])
        F1.append(F1_temp)
        e_F1.append(e_F1_temp)

        plot_graph(F1, e_F1, axis[i_task], task)

# set the values on x axis
for ax in axis:  
    plt.setp(ax, xticks=[1,2,5,10], xticklabels=x_labels)

# add last plot for legend
for i in range(len(models)): # need to add actual data so lengend shows
    ax6.plot(0,0,color=colors[i])       
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(colors))]
ax8.legend(handles, tasks,loc='center')

# make last plot invisible
ax8.spines['bottom'].set_color('#ffffff')
ax8.spines['top'].set_color('#ffffff') 
ax8.spines['right'].set_color('#ffffff')
ax8.spines['left'].set_color('#ffffff')
ax8.yaxis.label.set_color('#ffffff')
ax8.xaxis.label.set_color('#ffffff')
ax8.tick_params(axis='x', colors='#ffffff')
ax8.tick_params(axis='y', colors='#ffffff')



plt.show()