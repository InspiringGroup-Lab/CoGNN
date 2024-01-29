import numpy as np
from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import re

import statistics

from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# sns.set_style("white")
style_label = "default"

green = "mediumseagreen"
red = "salmon"
blue = "steelblue"

defaultNumParts = 2
defaultBandWidth = 400 #Mbps
defaultLatency = 1 #ms
defaultInterRatio = 0.4
defaultScaler = 0.2
bandwidthList = [200, 400, 1000, 4000]
latencyList = [0.15, 1, 10, 20]
interRatioList = [0.1, 0.2, 0.4, 0.8, 1]
numPartsList = [2,3,4,5]
scalerList = [0.05, 0.2, 0.5, 1]
executableList = ["sssp-ss", "pagerank-ss"]
executableNameList = ["(a) SSSP", "(b) PageRank"]

plot_data_path = "plot_data/formatted.txt"
            
def line_to_segs(line):
    line = line.replace(" ","")
    line = line.strip("\n")
    segs = [x for x in line.split("|") if x]
    return segs

def plot_executable_scale_bandwidth():
    with open(plot_data_path, "r", encoding="utf-8") as ifile:
        lines = ifile.readlines()
        begin_index = 0
        end_index = 0
        for i in range(len(lines)):
            if lines[i].startswith(">>>> Scale Bandwidth Figure"):
                begin_index = i+1
                break
        for i in range(begin_index, len(lines)):
            if lines[i].startswith(">>>> "):
                end_index = i - 1
                break
        i = begin_index
        print(begin_index, end_index)
        fig_data = {}
        while i <= end_index:
            # print("h", i)
            # print(i, lines[i])
            for executable in executableList:
                if executable in lines[i]:
                    fig_data[executable] = {}
                    segs = line_to_segs(lines[i])
                    fig_data[executable]["x"] = [int(x) for x in segs[1:]]
                    i += 1
                    for scaler in scalerList:
                        i += 1
                        segs = line_to_segs(lines[i])
                        assert segs[0] == str(scaler), f"segs[0] unequal scaler {segs[0]}, {scaler}"
                        fig_data[executable][segs[0]] = [int(x) for x in segs[1:]]
                    i += 1
        
        plt.style.use('seaborn-darkgrid')
        fig_size = [16, 3.5]
        fig, axes = plt.subplots(ncols=len(executableList), nrows=1, num=style_label,
                                figsize=fig_size, sharey=True)

        handles = []
        for i in range(len(executableList)):
            ax = axes[i]
            l_list = []
            for scaler in scalerList:
                cur_l, = ax.plot(fig_data[executableList[i]]["x"], fig_data[executableList[i]][str(scaler)], 'o', ls='-', ms=4, label=str(scaler))
                l_list.append(cur_l)
            if not handles:
                handles = l_list
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel('Bandwidth [Mbps]')
            if i == 0:
                ax.set_ylabel("Online Iteration Duration [s]")
            ax.set_title(executableNameList[i], pad=10)
        
        ph = [plt.plot([],marker="", ls="")[0]]
        handles = ph + handles
        fig.legend(handles, ["Graph Scale: "] + [str(scaler) for scaler in scalerList], loc='lower center',
                ncol=len(scalerList) + 1, fancybox=True, shadow=True)
        plt.subplots_adjust(left=0.05, bottom=0.3, right=0.98, top=0.9, wspace=0.2, hspace=0.4)
        fig.savefig("./fig/scale-bandwidth.pdf")

def plot_row_fig(header=">>>> Scale Bandwidth Figure", variableList=scalerList, legendTitle="Graph Scale: ", figFileName="scale-bandwidth.pdf"):
    with open(plot_data_path, "r", encoding="utf-8") as ifile:
        lines = ifile.readlines()
        begin_index = 0
        end_index = 0
        for i in range(len(lines)):
            if lines[i].startswith(header):
                begin_index = i+1
                break
        for i in range(begin_index, len(lines)):
            if lines[i].startswith(">>>> "):
                end_index = i - 1
                break
            elif i is (len(lines) - 1):
                end_index = i
                break
        i = begin_index
        print(begin_index, end_index)
        fig_data = {}
        while i <= end_index:
            # print("h", i)
            # print(i, lines[i])
            for executable in executableList:
                if executable in lines[i]:
                    fig_data[executable] = {}
                    segs = line_to_segs(lines[i])
                    fig_data[executable]["x"] = [int(x) for x in segs[1:]]
                    i += 1
                    for variable in variableList:
                        i += 1
                        segs = line_to_segs(lines[i])
                        assert segs[0] == str(variable), f"segs[0] unequal variable {segs[0]}, {variable}"
                        fig_data[executable][segs[0]] = [int(x) for x in segs[1:]]
                    i += 1
        
        plt.style.use('seaborn-darkgrid')
        fig_size = [16, 3.5]
        fig, axes = plt.subplots(ncols=len(executableList), nrows=1, num=style_label,
                                figsize=fig_size, sharey=True)

        handles = []
        for i in range(len(executableList)):
            ax = axes[i]
            l_list = []
            for variable in variableList:
                cur_l, = ax.plot(fig_data[executableList[i]]["x"], fig_data[executableList[i]][str(variable)], 'o', ls='-', ms=4, label=str(variable))
                l_list.append(cur_l)
            if not handles:
                handles = l_list
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel('Bandwidth [Mbps]')
            if i == 0:
                ax.set_ylabel("Online Iteration Duration [s]")
            ax.set_title(executableNameList[i], pad=10)
        
        ph = [plt.plot([],marker="", ls="")[0]]
        handles = ph + handles
        fig.legend(handles, [legendTitle] + [str(variable) for variable in variableList], loc='lower center',
                ncol=len(variableList) + 1, fancybox=True, shadow=True)
        plt.subplots_adjust(left=0.05, bottom=0.3, right=0.98, top=0.86, wspace=0.2, hspace=0.4)
        fig.savefig("./fig/"+figFileName)
        fig.clear()

def plot_new_row_fig(header=">>>> Scale Bandwidth Figure", variableList=scalerList, xLabel="Scaler", figFileName="scale-bandwidth.pdf"):
    with open(plot_data_path, "r", encoding="utf-8") as ifile:
        lines = ifile.readlines()
        begin_index = 0
        end_index = 0
        for i in range(len(lines)):
            if lines[i].startswith(header):
                begin_index = i+1
                break
        for i in range(begin_index, len(lines)):
            if lines[i].startswith(">>>> "):
                end_index = i - 1
                break
            elif i is (len(lines) - 1):
                end_index = i
                break
        i = begin_index
        print(begin_index, end_index)
        fig_data = {}
        while i <= end_index:
            # print("h", i)
            # print(i, lines[i])
            for executable in executableList:
                if executable in lines[i]:
                    fig_data[executable] = {}
                    segs = line_to_segs(lines[i])
                    fig_data[executable]["legend"] = [str(x) for x in segs[1:]]
                    i += 1
                    for variable in variableList:
                        i += 1
                        segs = line_to_segs(lines[i])
                        assert segs[0] == str(variable), f"segs[0] unequal variable {segs[0]}, {variable}"
                        fig_data[executable][segs[0]] = [int(x) for x in segs[1:]]
                    i += 1
        
        plot_data = {}
        for executable in executableList:
            plot_data[executable] = {}
            for i in range(len(fig_data[executable]["legend"])):
                cur_legend = fig_data[executable]["legend"][i]
                plot_data[executable][cur_legend] = []
                for variable in variableList:
                    plot_data[executable][cur_legend].append(fig_data[executable][str(variable)][i])
            plot_data[executable]['x'] = []
            for variable in variableList:
                plot_data[executable]['x'].append(float(variable))

        plt.style.use('seaborn-darkgrid')
        fig_size = [16, 3.5]
        fig, axes = plt.subplots(ncols=len(executableList), nrows=1, num=style_label,
                                figsize=fig_size, sharey=True)

        handles = []
        for i in range(len(executableList)):
            ax = axes[i]
            l_list = []
            for cur_legend in fig_data[executable]["legend"]:
                cur_l, = ax.plot(plot_data[executableList[i]]["x"], plot_data[executableList[i]][cur_legend], 'o', ls='-', ms=4, label=cur_legend)
                l_list.append(cur_l)
            if not handles:
                handles = l_list
            ax.yaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel(xLabel)
            if i == 0:
                ax.set_ylabel("Online Iteration Duration [s]")
            ax.set_title(executableNameList[i], pad=10)
        
        ph = [plt.plot([],marker="", ls="")[0]]
        handles = ph + handles
        fig.legend(handles, ["Bandwidth [Mbps]: "] + fig_data[executableList[0]]["legend"], loc='lower center',
                ncol=len(fig_data[executableList[0]]["legend"]) + 1, fancybox=True, shadow=True)
        plt.subplots_adjust(left=0.05, bottom=0.3, right=0.98, top=0.86, wspace=0.2, hspace=0.4)
        fig.savefig("./fig/"+figFileName)
        fig.clear()

def plot_part_new_row_fig(header=">>>> Scale Bandwidth Figure", variableList=scalerList, xLabel="Scaler", figFileName="scale-bandwidth.pdf"):
    executableList = ["sssp-ss", "pagerank-ss"]
    executableNameList = ["(a) SSSP", "(b) PageRank"]
    with open(plot_data_path, "r", encoding="utf-8") as ifile:
        lines = ifile.readlines()
        begin_index = 0
        end_index = 0
        for i in range(len(lines)):
            if lines[i].startswith(header):
                begin_index = i+1
                break
        for i in range(begin_index, len(lines)):
            if lines[i].startswith(">>>> "):
                end_index = i - 1
                break
            elif i is (len(lines) - 1):
                end_index = i
                break
        i = begin_index
        print(begin_index, end_index)
        fig_data = {}
        while i <= end_index:
            # print("h", i)
            # print(i, lines[i])
            # print(executableList)
            hasFlag = False
            for executable in executableList:
                if executable in lines[i]:
                    hasFlag = True
                    # print(lines[i])
                    fig_data[executable] = {}
                    segs = line_to_segs(lines[i])
                    fig_data[executable]["legend"] = [str(x) for x in segs[1:]]
                    i += 1
                    for variable in variableList:
                        i += 1
                        segs = line_to_segs(lines[i])
                        assert segs[0] == str(variable), f"segs[0] unequal variable {segs[0]}, {variable}"
                        fig_data[executable][segs[0]] = [int(x) for x in segs[1:]]
                    i += 1
            if not hasFlag:
                i += 1
        
        plot_data = {}
        for executable in executableList:
            plot_data[executable] = {}
            for i in range(len(fig_data[executable]["legend"])):
                cur_legend = fig_data[executable]["legend"][i]
                plot_data[executable][cur_legend] = []
                for variable in variableList:
                    plot_data[executable][cur_legend].append(fig_data[executable][str(variable)][i])
            plot_data[executable]['x'] = []
            for variable in variableList:
                plot_data[executable]['x'].append(float(variable))

        plt.style.use('seaborn-darkgrid')
        fig_size = [8, 3.5]
        fig, axes = plt.subplots(ncols=len(executableList), nrows=1, num=style_label,
                                figsize=fig_size, sharey=True)

        handles = []
        for i in range(len(executableList)):
            ax = axes[i]
            l_list = []
            for cur_legend in fig_data[executable]["legend"]:
                cur_l, = ax.plot(plot_data[executableList[i]]["x"], plot_data[executableList[i]][cur_legend], 'o', ls='-', ms=4, label=cur_legend)
                l_list.append(cur_l)
            if not handles:
                handles = l_list
            ax.yaxis.set_tick_params(labelbottom=True, rotation=45)
            ax.set_xlabel(xLabel)
            if i == 0:
                ax.set_ylabel("Online Iteration Duration [s]")
            ax.set_title(executableNameList[i], pad=10)
        
        ph = [plt.plot([],marker="", ls="")[0]]
        handles = ph + handles
        fig.legend(handles, ["Bandwidth [Mbps]:"] + fig_data[executableList[0]]["legend"], loc='lower center',
                ncol=len(fig_data[executableList[0]]["legend"]) + 1, fancybox=True, shadow=True, prop={'size': 14})
        plt.subplots_adjust(left=0.11, bottom=0.3, right=0.98, top=0.86, wspace=0.2, hspace=0.4)
        fig.savefig("./fig/"+figFileName)
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

# plot_new_row_fig()
# plot_new_row_fig(header=">>>> Inter-Ratio Bandwidth Figure", variableList=interRatioList, xLabel="Inter-Edge Ratio", figFileName="inter-ratio-bandwidth.pdf")
# plot_new_row_fig(header=">>>> Party Num Figure", variableList=numPartsList, xLabel="Party Number", figFileName="party-num-bandwidth.pdf")
# plot_new_row_fig(header=">>>> Latency Figure", variableList=latencyList, xLabel="Latency [ms]", figFileName="latency-bandwidth.pdf")

plot_part_new_row_fig()
plot_part_new_row_fig(header=">>>> Inter-Ratio Bandwidth Figure", variableList=interRatioList, xLabel="Inter-Edge Ratio", figFileName="inter-ratio-bandwidth.pdf")
plot_part_new_row_fig(header=">>>> Party Num Figure", variableList=numPartsList, xLabel="Party Number", figFileName="party-num-bandwidth.pdf")
# plot_part_new_row_fig(header=">>>> Latency Figure", variableList=latencyList, xLabel="Latency [ms]", figFileName="latency-bandwidth.pdf")