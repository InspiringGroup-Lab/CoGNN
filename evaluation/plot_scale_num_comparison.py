import numpy as np
from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import re
import math
import statistics

from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams.update({'font.size': 17})
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# sns.set_style("white")
style_label = "default"

green = "mediumseagreen"
red = "salmon"
blue = "steelblue"

defaultEVIndex = 3

def plot_comparison():
    EList = [1291676, 2599830, 6506882, 13027596]
    VList = [670436, 996528, 1388020, 1491258]

    numParty_EList = [2599830, 3899721, 5199660, 6499535]
    numParty_VList = [996528, 1494792, 1993056, 2491320]

    curScalerList = [0.1, 0.2, 0.5, 1]
    curNumPartyList = [2, 3, 4, 5]

    sysList = [6*EList[i]+2*VList[i] for i in range(len(EList))]
    outList = [2*(EList[i]+VList[i])*int(math.log(EList[i]+VList[i],2)) for i in range(len(EList))]

    numParty_sysList = [2*(6*numParty_EList[i]+curNumPartyList[i]*numParty_VList[i]) for i in range(len(numParty_EList))]
    numParty_outList = [curNumPartyList[i]*(2*(numParty_EList[i]+numParty_VList[i])*int(math.log(numParty_EList[i]+numParty_VList[i],2))) for i in range(len(numParty_EList))]

    print(numParty_sysList)
    print(numParty_outList)
    print([numParty_outList[i]/numParty_sysList[i] for i in range(len(numParty_sysList))])

    # plt.style.use('seaborn-darkgrid')
    fig_size = [9, 3.5]
    fig, axes = plt.subplots(ncols=2, nrows=1, num=style_label,
                            figsize=fig_size)

    handles = []

    cur_l, = axes[0].plot(curScalerList, sysList, 'd', ls='-', ms=4, color='steelblue', label="CoGNN")
    handles.append(cur_l)
    cur_l, = axes[0].plot(curScalerList, outList, '*', ls='-', ms=4, color='red', label="OGP")
    handles.append(cur_l)

    cur_l, axes[1].plot(curNumPartyList, numParty_sysList, 'd', ls='-', ms=4, color='steelblue', label="CoGNN")
    cur_l, axes[1].plot(curNumPartyList, numParty_outList, '*', ls='-', ms=4, color='red', label="OGP")

    axes[0].set_xlabel("Scaler")
    axes[0].set_ylabel("Online Total Work")

    axes[1].set_xlabel("# of Parties")
    # axes[1].set_ylabel("Online Iteration Total Work")

    # axes[1].legend(loc='upper left',
    #         fancybox=True, shadow=True)

    axes[0].set_title("(a)", pad=10)
    axes[1].set_title("(b)", pad=10)
    axes[0].yaxis.set_tick_params(labelbottom=True, rotation=45)
    axes[1].yaxis.set_tick_params(labelbottom=True, rotation=45)

    fig.legend(handles, ["CoGNN", "OGP"], loc='lower center',
            ncol=2, fancybox=True, shadow=True)

    plt.subplots_adjust(left=0.075, bottom=0.33, right=0.99, top=0.90, wspace=0.2, hspace=0.5)

    fig.savefig("./fig/sys-outsourced-comparison.pdf")
    fig.clear()


   



# def box_figure(ax):
#     global gen_tc_num, gt_tc_num
#     # delays = np.array([103,117,131,137,145,151,154,159,162,167,170,173,175,178,179,181,184])
#     # delays = delays / 100

#     # index = [x+1 for x in range(len(gt_tc_num))]
#     # ax.plot(index, gt_tc_num, marker='d', markersize=4 , color='steelblue')
#     # ax.plot(index, gen_tc_num, marker='*', markersize=4 , color='red')
    
#     # # ax.set_yticks(y_pos)
#     # ax.set_xticks(range(1,len(gt_tc_num) + 1,1))
#     # ax.set_xticklabels(range(1,len(gt_tc_num) + 1,1), fontsize=14)

#     tick_pos = []

#     # ax.bar(edit_num, width=1 , color='orange', linewidth=0.01, alpha=0.65)

#     edit_num_all = []
#     for x in edit_num:
#         edit_num_all += x

#     edit_num_all.sort()
#     print(edit_num_all)
#     print(np.median(edit_num_all))

#     ax.boxplot(edit_num, labels = cgs)
        
#     # ax.set_yticks(y_pos)
#     # ax.set_xticks(tick_pos)
#     # ax.set_xticklabels(cgs, fontsize=10)
#     # ax.tick_params(axis='both', which='both', length=0)

#     # ax.set_xticklabels(0,173,175,178,179,181,184])
#     ax.set_xlabel('Legal Agreement Category')
#     ax.set_ylabel("Manual Edit Num")
#     # ax.set_yscale('log')
#     # ax.set_title('Dispatch delay of different patch number', pad=10)
#     # ax.set_ylim(ymin=0, ymax=10)
#     # ax.set_xlabel("(a) Dispatch delay of various patch numbers") 
#     ax.xaxis.labelpad = 8
#     plt.sca(ax)
#     plt.xticks(rotation=10, fontsize=12)
#     # ax.legend(
#     #     loc='upper left', ncol=1)

# def draw_figure():
#     (fig_width, fig_height) = plt.rcParams['figure.figsize']
#     fig_size = [fig_width, fig_height / 1.5 ]
#     fig, axes = plt.subplots(ncols=1, nrows=1, num=style_label,
#                              figsize=fig_size, squeeze=True)
#     # axes[1].set_ylabel("Total Patch Delay (μs)")
#     box_figure(axes)
#     plt.subplots_adjust(left=0.08, bottom=0.24, right=0.97, top=0.96, wspace=0.23, hspace=0.4)
#     # plt.yscale("log")
#     fig.align_labels()
#     plt.show()
#     fig.savefig("RQ1-semantic.pdf")

# plot_executable_scale_bandwidth()
# plot_row_fig()
# plot_row_fig(header=">>>> Inter-Ratio Bandwidth Figure", variableList=interRatioList, legendTitle="Inter-Edge Ratio: ", figFileName="inter-ratio-bandwidth.pdf")
# plot_row_fig(header=">>>> Party Num Figure", variableList=numPartsList, legendTitle="Participant Number: ", figFileName="party-num-bandwidth.pdf")
# plot_row_fig(header=">>>> Latency Figure", variableList=latencyList, legendTitle="Latency [ms]: ", figFileName="latency-bandwidth.pdf")
# print_accuracy()

plot_comparison()