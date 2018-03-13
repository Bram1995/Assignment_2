import numpy as np
import copy
import cplex
import collections as col
import pandas as pd
from dijkstrasalgoritm import dijkstra, graph_creator


xl = pd.ExcelFile("Input_AE4424_Ass1P1.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
len(dfs['Commodities'].From.unique())
len(dfs['Commodities'].To.unique())

arcs = range(len(dfs['Arcs'].Arc))
origins = dfs['Arcs'].From
destinations = dfs['Arcs'].To
locations = pd.concat([dfs['Arcs'].From,dfs['Arcs'].To]).unique()
commodities = range(1, len(dfs['Commodities'].Commodity) + 1)
costs = dfs['Arcs'].Cost

#initiation algorithm
graph = graph_creator()

#produce inequality matrix A_ineq
A_ineq = np.zeros((len(arcs),len(commodities)))
rhs= np.zeros((len(arcs),1))
for i in range(len(commodities)):
    path, path_cost = dijkstra(graph, dfs['Commodities'].From[i], dfs['Commodities'].To[i])
    for j in range(len(path)-1):
        index = dfs['Arcs'].index[(dfs['Arcs'].From == path[j]) & (dfs['Arcs'].To == path[j+1])]
        A_ineq[index,i] = 1*dfs['Commodities'].Quant[i]
        rhs[index] = dfs['Arcs'].Capacity[index]

A_ineq_Sum = np.sum(A_ineq,axis=1)

#produce slack columns with labels
Slack_column_labels=[]
for i in range(len(arcs)):
    Slack_column = np.zeros((len(arcs), 1))
    s = A_ineq_Sum[i]-rhs[i]
    if s > 0:
        Slack_column[i]=-1
        A_ineq = np.c_[A_ineq,Slack_column]
        Slack_column_labels.append(['s_' + dfs['Arcs'].From[i] + '_' + dfs['Arcs'].To[i]])


#adjust graph with new costs
dual_costs_array=np.ones(len(arcs)) #example array
original_graph=copy.deepcopy(graph)
for i in range(len(arcs)):
    origin = dfs['Arcs'].From[i]
    destination = dfs['Arcs'].To[i]
    graph[origin][destination]=original_graph[origin][destination]+dual_costs_array[i]

