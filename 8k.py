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


if __name__ == "__main__":
    '''
    tp = 8
    dp = 64
    pp = 16
    ep = 1
    '''
    tp = 32
    dp = 8
    pp = 8
    ep = 4
    npu= tp*dp*pp
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

    print "=======================DP start2"
    print('\n'.join(' '.join(map(str,sl)) for sl in DP))
    print len(DP)

    for sl in DP:
        for i in sl:
            print i/64,
        print

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
    print "=======================EP start2"

    for sl in EP:
        for i in sl:
            print i/64,
        print

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
    for sl in PP:
        for i in sl:
            print i/64,
        print
    print "=======================PP over2"

