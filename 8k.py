#!/usr/bin/env python
# coding: utf-8

import sys
import hashlib
import math
import random
import string
import matplotlib.pyplot as plt
from operator import itemgetter
from pprint import pprint
import numpy as np


def to_rack(dd):
    for sl in dd:
        for i in sl:
            print i/64,
        print

if __name__ == "__main__":
    '''
    tp = 8
    dp = 64
    pp = 16
    ep = 1
    '''
    '''
    tp = 32
    dp = 8
    pp = 8
    ep = 4
    '''
    tp = 32
    dp = 32
    pp = 1
    ep = 8
    npu= tp*dp*pp

    rack_num= npu/64
    #npu = 64
    #tp = 8
    #dp = 4
    #pp = 2

    old_stdout = sys.stdout

    tp_log_file = open("tp.log","w")
    pp_log_file = open("pp.log","w")
    dp_log_file = open("dp.log","w")
    ep_log_file = open("ep.log","w")

    TP=[]
    DP=[]
    PP=[]
    tmp=[]

    comm_graph={}

    for x in range(rack_num):
        for y in range(rack_num):
            comm_graph[str(x)+'_'+str(y)] = {'dp':0, 'ep':0, 'sp':0, 'pp':0, 'all':0, 'tp':0}

    #new_comm_graph=dict(sorted(comm_graph.items()))
    #print comm_graph

    sys.stdout = tp_log_file
    print "=======================TP start1"
    for j in range(npu):
        #wyl
        if j%tp !=0: 
            tmp.append(j)
        else:
            if len(tmp) != 0:
                TP.append(tmp)
            tmp=[j]
        #print tmp
        #print TP

    if len(tmp) != 0:
        TP.append(tmp)

    print TP, len(TP)

    print "=======================TP over"
    
    sys.stdout = dp_log_file
    print "=======================DP start1"
    for i in range(tp):
        tmp=[]
        n=0
        for j in TP:
            n+=1
            tmp.append(j[i])
            if n%dp == 0: 
                DP.append(tmp)
                tmp=[]
    #print DP, len(DP)

    DP.sort(key=lambda x: x[0])

    #pprint(DP)

    print('\n'.join(' '.join(map(str,sl)) for sl in DP))
    print len(DP)

    print "=======================DP start2"

    to_rack(DP)

    #ring
    for sl in DP:
        for i in range(dp):
            source = i
            dest = (i+1)%dp
            key=str(sl[source]/64) + '_' + str(sl[dest]/64)
            comm_graph[key]['dp'] +=1


    print comm_graph
    print "=======================DP over"

    #sp=2
    SP=[]
    for i in DP:
        SP.append(i[0:32])
        SP.append(i[32:64])

    print('\n'.join(' '.join(map(str,sl)) for sl in SP))
    print len(SP)
    print "=======================SP over"
    sys.stdout = ep_log_file
    print "=======================EP start1"

    EP=[]
    for i in DP:
        scale=dp/ep
        for n in range(scale):
            EP.append(i[n*ep:(n+1)*ep])

    print('\n'.join(' '.join(map(str,sl)) for sl in EP))
    print len(SP)

    #tmp
    for sl in EP:
        if sl[0]<64:
            print sl

    print "=======================EP start2"
    to_rack(EP)
    #mesh
    for sl in EP:
        for x in range(ep):
            for y in range(ep):
                source = x
                dest = y
                key=str(sl[source]/64) + '_' + str(sl[dest]/64)
                comm_graph[key]['ep'] +=1

    print comm_graph
    print "=======================EP over"


    sys.stdout = pp_log_file
    print "=======================PP start1"
    pace=len(DP)/pp

    for m in range(pace):
        for i in range(dp):
            tmp=[]
            n=0
            n2=0
            for j in DP:
                if n%pace == m:
                    tmp.append(j[i])
                #print n,tmp
                n+=1
                if len(tmp) == pp:
                    PP.append(tmp)
                    tmp=[]
        #print PP, len(PP)

    print('\n'.join(' '.join(map(str,sl)) for sl in PP))
    print len(PP)

    print "=======================PP over1"
    to_rack(PP)
    #ring
    for sl in PP:
        for i in range(pp):
            if i == (pp-1):
                break
            source = i
            dest = (i+1)
            key=str(sl[source]/64) + '_' + str(sl[dest]/64)
            comm_graph[key]['pp'] +=1

    print comm_graph
    print "=======================PP over2"

    sys.stdout = old_stdout

    comm_link_max = 0
    for sl in comm_graph:
        comm_graph[sl]['all'] = comm_graph[sl]['pp'] + comm_graph[sl]['tp'] + comm_graph[sl]['ep'] + comm_graph[sl]['dp']
        tmp=sl.split("_")
        #just get link between rack
        if comm_graph[sl]['all'] != 0 and tmp[0] != tmp[1]:
            print sl,comm_graph[sl]['all']
            comm_link_max += comm_graph[sl]['all']

    #print comm_graph
    data = np.zeros((rack_num, rack_num))
    
    for x in range(rack_num):
        for y in range(rack_num):
            tmp=comm_graph[str(x)+'_'+str(y)]['all']
            data[x,y] = tmp
            if x != y and tmp != 0:
                print x,y,float(comm_graph[str(x)+'_'+str(y)]['all'] * 32*224)/ comm_link_max
            
    #print data
    plt.imshow(data, cmap='hot')
    plt.colorbar()
    plt.show()
