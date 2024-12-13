from typing import Any, DefaultDict, List
from networkx.classes.graph import Graph
from heapq import heappop, heappush
from collections import defaultdict
import networkx as nx
import multiprocessing

def restore_path(source: Any, target: Any, prev: DefaultDict):
    path = []
    last = target
    while True:
        path.append(last)
        last = prev[last]
        if last is None:break
    if path == []:
        return -1
    elif source != path[-1]:
        return -1
    return path[::-1]

def dijkstra_with_builtin_heap(adjascent: List[dict], source: Any, target: Any, edge_weight, ignore_node=None, ignore_edge=None):
    adjascent = adjascent[0]
    dist = defaultdict(lambda: float('inf'))
    prev = defaultdict(lambda: None)
    visited = defaultdict(lambda: False)
    if ignore_edge is None:ignore_edge=set()
    if ignore_node is None:ignore_node=set()
    dist[source] = 0
    q = [(0,source)]
    while q:
        cur = heappop(q)
        cost, node = cur
        #print("------", cost, node)
        if visited[node]:continue
        visited[node] = True
        for adj in adjascent[node]:
            if adj in ignore_node:continue
            elif (node, adj) in ignore_edge:continue
            if dist[adj] > cost + edge_weight[(node, adj)]:
                dist[adj] = cost + edge_weight[(node, adj)]
                heappush(q,(dist[adj],adj))
                prev[adj] = node
    Path = restore_path(source, target, prev)
    return dist[target], Path

def bidirectional_dijkstra_with_builtin_heap(adjascent: List[dict], source: Any, target: Any, edge_weight, ignore_node=None, ignore_edge=None):
    dist = [defaultdict(lambda: float('inf')), defaultdict(lambda: float('inf'))]
    path = [{source:[source]}, {target:[target]}]
    visited = [defaultdict(lambda: False), defaultdict(lambda: False)]
    if ignore_edge is None:ignore_edge=set()
    if ignore_node is None:ignore_node=set()
    dist[0][source] = 0
    dist[1][target] = 0
    q = [[(0,source)],[(0,target)]]
    dir = 1
    finalPath = []
    finalDist = float('inf')
    while q[0] and q[1]:
        dir = 1-dir
        cur = heappop(q[dir])
        cost, node = cur
        if visited[dir][node]:continue
        visited[dir][node] = True
        if visited[1-dir][node]:
            
            return finalDist, finalPath
        #print("11 ", "dir=", dir, 'node=', node, adjascent[dir])
        for adj in adjascent[dir][node]:
            #print("adj=", adj, "dir=", dir, 'node=', node, adjascent)
            if adj in ignore_node:continue
            if dir == 0:e = (node, adj)
            else: e = (adj,node)
            if e in ignore_edge:continue
            if dist[dir][adj] > cost + edge_weight[e]:
                dist[dir][adj] = cost + edge_weight[e]
                heappush(q[dir],(dist[dir][adj],adj))
                path[dir][adj] = path[dir][node] + [adj]
            if dist[0][adj] < float('inf') and dist[1][adj] < float('inf'):
                totalDist = dist[0][adj]+dist[1][adj]
                if finalPath == [] or finalDist > totalDist:
                    finalDist = totalDist
                    finalPath = path[0][adj] + path[1][adj][::-1][1:]

    return dist[target], path

def construct(G: Graph, weight="weight"):
    e_weight = {}
    adj, radj = {}, {}
    edges = G.edges()
    for e in edges:
        #print('e=', e)
        e_weight[e] = edges[e][weight]
        if e[0] not in adj:
            adj[e[0]] = []
        adj[e[0]].append(e[1])
        #print(adj)
        if e[1] not in radj:
            radj[e[1]] = []
        radj[e[1]].append(e[0])

    #print("adj=",adj)
    #print("radj=",radj)
    #print("e_weight=",e_weight)
    return e_weight, [adj, radj]

def yenksp(G: Graph, source: Any, target: Any, k: int, weight: str ="weight", shortest_path_func=bidirectional_dijkstra_with_builtin_heap):
    listA = []
    listB = PathBuffer()
    prev_path = None
    edge_weight, adjascent = construct(G, weight=weight)
    for _ in range(k):
        if prev_path is None:
            print("prev_path None=================")
            length, path = shortest_path_func(adjascent, source, target, edge_weight)
            if isinstance(path, list):
                listB.push(length, path)
            else:
                raise NotImplementedError(f"No path exists between node {source} and node {target}.")
        else:
            ignore_nodes = set()

            ss=(source)//128
            tt=(target)//128
            #print("ss", ss, "tt", tt);

            # if conn between different rack, ignore Dnode path in same rack
            for node in [source, target]:
                if ss!=tt:
                    D_range=range(node//128*128,  node//128*128+num_D_node)
                    if node in D_range:
                        tmp = G.in_edges(node)
                        for xx in tmp:
                            #print("xx ", node, xx)
                            if xx[0] in D_range and xx[1] in D_range:
                                for sl in xx:
                                    if sl != node and sl not in ignore_nodes:
                                        #print("add ", sl)
                                        ignore_nodes.add(sl)

            ignore_edges = set()
            #reserve only 1 path between L1 L2 in other racks
            for nrank in range(0, rack_num):
                L1_start=nrank*num_per_rack_node + num_D_node
                L2_start=nrank*num_per_rack_node + num_D_node + num_l1_union_node

                for nr in range(0, rail):
                    for x in range(L1_start+ nr*d_columns, L1_start+d_columns*(nr+1)):
                        for y in range(L2_start + nr*d_columns, L2_start + d_columns*(nr+1)):
                            #ignore source down edge
                            if nrank == ss:
                                ignore_edges.add((y,x))
                            #ignore dest up edge
                            elif nrank == tt:
                                ignore_edges.add((x,y))
                            #reserve only 1 path between L1 L2 in other racks
                            #elif y%128 != 96 and x%128 !=64:
                            #elif x%128 !=64 and y%128 !=96:
                            elif x%128 not in [64,72,80,88]:
                                ignore_edges.add((y,x))
                                ignore_edges.add((x,y))
                                #print("l1 l2 ignore", x, y)

                        # ignore all other ranks top D node
                        if nrank not in [ss, tt]:
                            tmp = G.in_edges(x)
                            for sl in tmp:
                                for sy in sl:
                                    if sy != x and sy not in range(L2_start + nr*d_columns, L2_start + d_columns*(nr+1)) and sy not in ignore_nodes:
                                        ignore_nodes.add(sy)


            # remove add EP mesh routing

                            
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                #print(i, "root", root)
                root_length = sum([edge_weight[(u,v)] for u,v in zip(root, root[1:])])
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i-1],path[i]))
                        #print(i,"ignore edges", path[i-1],path[i] )
                #print(i, "root ignore", ignore_edges, ignore_nodes)

                #print(i, "compute", root[-1], target)
                length, supr = shortest_path_func(adjascent, root[-1], target, edge_weight, ignore_node=ignore_nodes, ignore_edge=ignore_edges)
                #print(i, "get", supr)
                if isinstance(supr, list):
                    #remove EP mesh routing
                    remove=0
                    '''
                    for n in range(len(supr)):
                        if n < len(supr)-1:
                            sn=supr[n]//128
                            sn1=supr[n+1]//128
                            if sn != sn1 and sn//4 == sn1//4:
                                # TODO: just print here now
                                #remove=1
                                print('n', n, sn, sn1, sn//4, sn1//4)
                    '''

                    #length jump limit
                    if remove ==0 and len(supr) + len(root[:-1]) <= 9:
                        listB.push(root_length+length, root[:-1]+supr)
                        #print(i, 'listB===>', listB.sortedpaths)
                ignore_nodes.add(root[-1])

        if listB:
            listA.append(listB.pop())
            prev_path = listA[-1]
            #print("update listA", listA)
        else:
            break
    return listA



class PathBuffer:
    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path in self.paths:return
        heappush(self.sortedpaths, (cost, path))
        self.paths.add(hashable_path)

    def pop(self):
        _, path = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path


def esp(G: Graph, test_cases, weight: str ="weight", shortest_path_func=bidirectional_dijkstra_with_builtin_heap):
        #for test in test_cases:
        print("iteration -=============************************************>", iteraton)
        path_log[str(test[0])+'_'+str(test[1])]=[]
        ret=yenksp(G, test[0], test[1], test[2], shortest_path_func=dijkstra_with_builtin_heap)
        return ret

# simple way
#G  = nx.Graph()
G  = nx.DiGraph()

# 添加节点

num_D_node=64
d_row = 8
d_columns = 8
num_l1_union_node =8
num_l2_union_node =8

num_node= num_D_node + num_l1_union_node + num_l2_union_node

for i in range(num_node):
    G.add_node(i)

# 添加边

# 横排Ｄ
for l in range(1, d_columns+1):
    for x in range(d_row*(l-1), d_row*l):
        for y in range(d_row*(l-1), d_row*l):
            if x!=y:
                G.add_edge(x, y, weight=1)

# 纵排Ｄ
for l in range(0, d_columns):
    union =  num_D_node + l
    for x in range(0+l, d_columns*d_row+l, d_columns):
        # L2 union 
        G.add_edge(x, union, weight=1)
        G.add_edge(union, x, weight=1)
        for y in range(0+l, d_columns*d_row+l, d_columns):
            if x!=y:
                G.add_edge(x, y, weight=1)


for x in range(num_D_node, num_D_node+num_l1_union_node):
    for y in range(num_D_node+num_l1_union_node, num_D_node+num_l1_union_node+num_l2_union_node):
        G.add_edge(x, y, weight=1)
        G.add_edge(y, x, weight=1)


# standord way
num_D_node=64
d_row = 8
d_columns = 8
rail=4
num_l1_union_node =8*rail  #4 rail
num_l2_union_node =8*rail

num_node= num_D_node + num_l1_union_node + num_l2_union_node
num_per_rack_node=num_node

rack_num=32
num_all_node= num_per_rack_node * rack_num
node_list=[]

G2  = nx.DiGraph()
for i in range(num_all_node):
    G2.add_node(i)
    node_list.append(i)

index1=0
index2=0

test_num_list=[1248,64,0]


#rack 
# 0~63  Dnode
# L1 rail0 64,65,...,71
# L1 rail1 72,73,...,79
# L1 rail2 80,81,...,87
# L1 rail3 88,89,...,95
#for range_rack_n in range(0, num_all_node, num_per_rack_node):
for n in range(rack_num):
    current_rack_node_range=node_list[(n)*num_per_rack_node:(n+1)*num_per_rack_node]
    print("rack ", n, current_rack_node_range)

    # 横排 mesh
    for l in range(1, d_columns+1):
        for x in range(d_row*(l-1), d_row*l):
            for y in range(d_row*(l-1), d_row*l):
                if x!=y:
                   G2.add_edge(x+n*num_per_rack_node, y+n*num_per_rack_node, weight=1)

    #for test_num in test_num_list:
    #    print("test1", test_num, G2.in_edges(test_num), len(G2.in_edges(test_num)))

    # 纵排Ｄ mesh and connect L1 union
    for nr in range(0, rail):
        for l in range(0, d_columns):
            #print(num_D_node, l, nr, d_columns)
            union =  num_D_node + l + nr*d_columns
            for x in range(0+l, d_columns*d_row+l, d_columns):
                # L2 union
                #print("conn", x+ n*num_per_rack_node, union+ n*num_per_rack_node, union, l, x)
                G2.add_edge(x+ n*num_per_rack_node, union+ n*num_per_rack_node, weight=1)
                G2.add_edge(union + n*num_per_rack_node , x + n*num_per_rack_node, weight=1)
                for y in range(0+l, d_columns*d_row+l, d_columns):
                    if x!=y:
                        G2.add_edge(x+ n*num_per_rack_node, y + n*num_per_rack_node, weight=1)
    #for test_num in test_num_list:
    #    print("test1", test_num, G2.in_edges(test_num), len(G2.in_edges(test_num)))

    # mesh between L1 and L2 union
    for nr in range(0, rail):
        for x in range(num_D_node + nr*d_columns, num_D_node+d_columns*(nr+1)):
            for y in range(num_D_node+ num_l1_union_node + nr*d_columns, num_D_node+ num_l1_union_node + d_columns*(nr+1)):
                #print("conn", x, y)
                G2.add_edge(x + n*num_per_rack_node, y + n*num_per_rack_node, weight=1)
                G2.add_edge(y + n*num_per_rack_node, x + n*num_per_rack_node, weight=1)

    #for test_num in test_num_list:
    #    print("test1", test_num, G2.in_edges(test_num), len(G2.in_edges(test_num)))

    # 2D between rack for O
    L2_union_range=range(num_D_node+ num_l1_union_node, num_D_node+ num_l1_union_node + num_l2_union_node)
    L2_union_start=num_D_node+ num_l1_union_node

    index1=n
    index2=n
    for m in range(n+1, rack_num):
        if n!= m:
            xx=L2_union_start + index1 + n*num_per_rack_node
            yy=L2_union_start + index2 + m*num_per_rack_node
            G2.add_edge(xx, yy, weight=1)
            G2.add_edge(yy, xx, weight=1)

            print("conn rack", (xx)//128, xx%128, "to", (int)(yy)//128, yy%128, '| id', xx, yy, "| index1", index1, "index2", index2, m, n)
            '''
            index1 = (index1+1)% num_l2_union_node
            index2 = (index2)% num_l2_union_node
            '''
            #we should use the same rail, like rail0 in L318
            print(index1, index2, m, num_l2_union_node)
            index1 = (index1+1) % 8
            index2 = (index2) % 8
            print(index1, index2, m, num_l2_union_node)

    '''
            for nrank in range(0, rack_num):
                L1_start=nrank*num_per_rack_node + num_D_node
                L2_start=nrank*num_per_rack_node + num_D_node + num_l1_union_node

                for nr in range(0, rail):
                    for x in range(L1_start+ nr*d_columns, L1_start+d_columns*(nr+1)):
                        for y in range(L2_start + nr*d_columns, L2_start + d_columns*(nr+1)):
    '''
    # TODO, 1.5D

for test_num in test_num_list:
    print("test1", test_num, G2.in_edges(test_num), len(G2.in_edges(test_num)))

'''
# 查看图的节点
print("图的节点:", G.nodes())

# 查看图的边
print("图的边:", G.edges())

# 获取节点的邻居

for i in range(num_D_node):
    print("节点", i,"的邻居:", list(G.neighbors(i)), len(list(G.neighbors(i))))
    #print("节点23的邻居:", list(G.neighbors(23)), len(list(G.neighbors(23))))

shortest_path = nx.shortest_path(G, source=1, target=5, weight='weight')
print("节点1到节点5的最短路径：", shortest_path)

print(G.has_edge(1, 65))  
print(G.has_edge(65, 1))  
'''

#for e in G.edges():
#    print(e)
'''
print("=====================")
ret=yenksp(G, 1, 9, 10, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
ret=yenksp(G, 2, 73, 11, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
ret=yenksp(G2, 1, 9, 10, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
#ret=yenksp(G2, 1, 9, 10)
#print(ret)
ret=yenksp(G2, 2, 73, 11, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
ret=yenksp(G2, 2, 1024, 11, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
ret=yenksp(G2, 0, 128, 11, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
'''
#ret=yenksp(G2, 0, 2*128+1, 33, shortest_path_func=dijkstra_with_builtin_heap)
#print("=====================", len(ret))

test_cases=[(0, 2*128+1, 31), 
           #(1,9,10),
            ]

'''
    tp = 32
    dp = 32
    pp = 2
    ep = 8
'''
ep_comm=[
[0,32,64,96,128,160,192,224],
[256,288,320,352,384,416,448,480],
[512,544,576,608,640,672,704,736],
[768,800,832,864,896,928,960,992],
[1,33,65,97,129,161,193,225],
[257,289,321,353,385,417,449,481],
[513,545,577,609,641,673,705,737],
[769,801,833,865,897,929,961,993],
[2,34,66,98,130,162,194,226],
[258,290,322,354,386,418,450,482],
[514,546,578,610,642,674,706,738],
[770,802,834,866,898,930,962,994],
[3,35,67,99,131,163,195,227],
[259,291,323,355,387,419,451,483],
[515,547,579,611,643,675,707,739],
[771,803,835,867,899,931,963,995],
[4,36,68,100,132,164,196,228],
[260,292,324,356,388,420,452,484],
[516,548,580,612,644,676,708,740],
[772,804,836,868,900,932,964,996],
[5,37,69,101,133,165,197,229],
[261,293,325,357,389,421,453,485],
[517,549,581,613,645,677,709,741],
[773,805,837,869,901,933,965,997],
[6,38,70,102,134,166,198,230],
[262,294,326,358,390,422,454,486],
[518,550,582,614,646,678,710,742],
[774,806,838,870,902,934,966,998],
[7,39,71,103,135,167,199,231],
[263,295,327,359,391,423,455,487],
[519,551,583,615,647,679,711,743],
[775,807,839,871,903,935,967,999],
[8,40,72,104,136,168,200,232],
[264,296,328,360,392,424,456,488],
[520,552,584,616,648,680,712,744],
[776,808,840,872,904,936,968,1000],
[9,41,73,105,137,169,201,233],
[265,297,329,361,393,425,457,489],
[521,553,585,617,649,681,713,745],
[777,809,841,873,905,937,969,1001],
[10,42,74,106,138,170,202,234],
[266,298,330,362,394,426,458,490],
[522,554,586,618,650,682,714,746],
[778,810,842,874,906,938,970,1002],
[11,43,75,107,139,171,203,235],
[267,299,331,363,395,427,459,491],
[523,555,587,619,651,683,715,747],
[779,811,843,875,907,939,971,1003],
[12,44,76,108,140,172,204,236],
[268,300,332,364,396,428,460,492],
[524,556,588,620,652,684,716,748],
[780,812,844,876,908,940,972,1004],
[13,45,77,109,141,173,205,237],
[269,301,333,365,397,429,461,493],
[525,557,589,621,653,685,717,749],
[781,813,845,877,909,941,973,1005],
[14,46,78,110,142,174,206,238],
[270,302,334,366,398,430,462,494],
[526,558,590,622,654,686,718,750],
[782,814,846,878,910,942,974,1006],
[15,47,79,111,143,175,207,239],
[271,303,335,367,399,431,463,495],
[527,559,591,623,655,687,719,751],
[783,815,847,879,911,943,975,1007],
[16,48,80,112,144,176,208,240],
[272,304,336,368,400,432,464,496],
[528,560,592,624,656,688,720,752],
[784,816,848,880,912,944,976,1008],
[17,49,81,113,145,177,209,241],
[273,305,337,369,401,433,465,497],
[529,561,593,625,657,689,721,753],
[785,817,849,881,913,945,977,1009],
[18,50,82,114,146,178,210,242],
[274,306,338,370,402,434,466,498],
[530,562,594,626,658,690,722,754],
[786,818,850,882,914,946,978,1010],
[19,51,83,115,147,179,211,243],
[275,307,339,371,403,435,467,499],
[531,563,595,627,659,691,723,755],
[787,819,851,883,915,947,979,1011],
[20,52,84,116,148,180,212,244],
[276,308,340,372,404,436,468,500],
[532,564,596,628,660,692,724,756],
[788,820,852,884,916,948,980,1012],
[21,53,85,117,149,181,213,245],
[277,309,341,373,405,437,469,501],
[533,565,597,629,661,693,725,757],
[789,821,853,885,917,949,981,1013],
[22,54,86,118,150,182,214,246],
[278,310,342,374,406,438,470,502],
[534,566,598,630,662,694,726,758],
[790,822,854,886,918,950,982,1014],
[23,55,87,119,151,183,215,247],
[279,311,343,375,407,439,471,503],
[535,567,599,631,663,695,727,759],
[791,823,855,887,919,951,983,1015],
[24,56,88,120,152,184,216,248],
[280,312,344,376,408,440,472,504],
[536,568,600,632,664,696,728,760],
[792,824,856,888,920,952,984,1016],
[25,57,89,121,153,185,217,249],
[281,313,345,377,409,441,473,505],
[537,569,601,633,665,697,729,761],
[793,825,857,889,921,953,985,1017],
[26,58,90,122,154,186,218,250],
[282,314,346,378,410,442,474,506],
[538,570,602,634,666,698,730,762],
[794,826,858,890,922,954,986,1018],
[27,59,91,123,155,187,219,251],
[283,315,347,379,411,443,475,507],
[539,571,603,635,667,699,731,763],
[795,827,859,891,923,955,987,1019],
[28,60,92,124,156,188,220,252],
[284,316,348,380,412,444,476,508],
[540,572,604,636,668,700,732,764],
[796,828,860,892,924,956,988,1020],
[29,61,93,125,157,189,221,253],
[285,317,349,381,413,445,477,509],
[541,573,605,637,669,701,733,765],
[797,829,861,893,925,957,989,1021],
[30,62,94,126,158,190,222,254],
[286,318,350,382,414,446,478,510],
[542,574,606,638,670,702,734,766],
[798,830,862,894,926,958,990,1022],
[31,63,95,127,159,191,223,255],
[287,319,351,383,415,447,479,511],
[543,575,607,639,671,703,735,767],
[799,831,863,895,927,959,991,1023],
[1024,1056,1088,1120,1152,1184,1216,1248],
[1280,1312,1344,1376,1408,1440,1472,1504],
[1536,1568,1600,1632,1664,1696,1728,1760],
[1792,1824,1856,1888,1920,1952,1984,2016],
[1025,1057,1089,1121,1153,1185,1217,1249],
[1281,1313,1345,1377,1409,1441,1473,1505],
[1537,1569,1601,1633,1665,1697,1729,1761],
[1793,1825,1857,1889,1921,1953,1985,2017],
[1026,1058,1090,1122,1154,1186,1218,1250],
[1282,1314,1346,1378,1410,1442,1474,1506],
[1538,1570,1602,1634,1666,1698,1730,1762],
[1794,1826,1858,1890,1922,1954,1986,2018],
[1027,1059,1091,1123,1155,1187,1219,1251],
[1283,1315,1347,1379,1411,1443,1475,1507],
[1539,1571,1603,1635,1667,1699,1731,1763],
[1795,1827,1859,1891,1923,1955,1987,2019],
[1028,1060,1092,1124,1156,1188,1220,1252],
[1284,1316,1348,1380,1412,1444,1476,1508],
[1540,1572,1604,1636,1668,1700,1732,1764],
[1796,1828,1860,1892,1924,1956,1988,2020],
[1029,1061,1093,1125,1157,1189,1221,1253],
[1285,1317,1349,1381,1413,1445,1477,1509],
[1541,1573,1605,1637,1669,1701,1733,1765],
[1797,1829,1861,1893,1925,1957,1989,2021],
[1030,1062,1094,1126,1158,1190,1222,1254],
[1286,1318,1350,1382,1414,1446,1478,1510],
[1542,1574,1606,1638,1670,1702,1734,1766],
[1798,1830,1862,1894,1926,1958,1990,2022],
[1031,1063,1095,1127,1159,1191,1223,1255],
[1287,1319,1351,1383,1415,1447,1479,1511],
[1543,1575,1607,1639,1671,1703,1735,1767],
[1799,1831,1863,1895,1927,1959,1991,2023],
[1032,1064,1096,1128,1160,1192,1224,1256],
[1288,1320,1352,1384,1416,1448,1480,1512],
[1544,1576,1608,1640,1672,1704,1736,1768],
[1800,1832,1864,1896,1928,1960,1992,2024],
[1033,1065,1097,1129,1161,1193,1225,1257],
[1289,1321,1353,1385,1417,1449,1481,1513],
[1545,1577,1609,1641,1673,1705,1737,1769],
[1801,1833,1865,1897,1929,1961,1993,2025],
[1034,1066,1098,1130,1162,1194,1226,1258],
[1290,1322,1354,1386,1418,1450,1482,1514],
[1546,1578,1610,1642,1674,1706,1738,1770],
[1802,1834,1866,1898,1930,1962,1994,2026],
[1035,1067,1099,1131,1163,1195,1227,1259],
[1291,1323,1355,1387,1419,1451,1483,1515],
[1547,1579,1611,1643,1675,1707,1739,1771],
[1803,1835,1867,1899,1931,1963,1995,2027],
[1036,1068,1100,1132,1164,1196,1228,1260],
[1292,1324,1356,1388,1420,1452,1484,1516],
[1548,1580,1612,1644,1676,1708,1740,1772],
[1804,1836,1868,1900,1932,1964,1996,2028],
[1037,1069,1101,1133,1165,1197,1229,1261],
[1293,1325,1357,1389,1421,1453,1485,1517],
[1549,1581,1613,1645,1677,1709,1741,1773],
[1805,1837,1869,1901,1933,1965,1997,2029],
[1038,1070,1102,1134,1166,1198,1230,1262],
[1294,1326,1358,1390,1422,1454,1486,1518],
[1550,1582,1614,1646,1678,1710,1742,1774],
[1806,1838,1870,1902,1934,1966,1998,2030],
[1039,1071,1103,1135,1167,1199,1231,1263],
[1295,1327,1359,1391,1423,1455,1487,1519],
[1551,1583,1615,1647,1679,1711,1743,1775],
[1807,1839,1871,1903,1935,1967,1999,2031],
[1040,1072,1104,1136,1168,1200,1232,1264],
[1296,1328,1360,1392,1424,1456,1488,1520],
[1552,1584,1616,1648,1680,1712,1744,1776],
[1808,1840,1872,1904,1936,1968,2000,2032],
[1041,1073,1105,1137,1169,1201,1233,1265],
[1297,1329,1361,1393,1425,1457,1489,1521],
[1553,1585,1617,1649,1681,1713,1745,1777],
[1809,1841,1873,1905,1937,1969,2001,2033],
[1042,1074,1106,1138,1170,1202,1234,1266],
[1298,1330,1362,1394,1426,1458,1490,1522],
[1554,1586,1618,1650,1682,1714,1746,1778],
[1810,1842,1874,1906,1938,1970,2002,2034],
[1043,1075,1107,1139,1171,1203,1235,1267],
[1299,1331,1363,1395,1427,1459,1491,1523],
[1555,1587,1619,1651,1683,1715,1747,1779],
[1811,1843,1875,1907,1939,1971,2003,2035],
[1044,1076,1108,1140,1172,1204,1236,1268],
[1300,1332,1364,1396,1428,1460,1492,1524],
[1556,1588,1620,1652,1684,1716,1748,1780],
[1812,1844,1876,1908,1940,1972,2004,2036],
[1045,1077,1109,1141,1173,1205,1237,1269],
[1301,1333,1365,1397,1429,1461,1493,1525],
[1557,1589,1621,1653,1685,1717,1749,1781],
[1813,1845,1877,1909,1941,1973,2005,2037],
[1046,1078,1110,1142,1174,1206,1238,1270],
[1302,1334,1366,1398,1430,1462,1494,1526],
[1558,1590,1622,1654,1686,1718,1750,1782],
[1814,1846,1878,1910,1942,1974,2006,2038],
[1047,1079,1111,1143,1175,1207,1239,1271],
[1303,1335,1367,1399,1431,1463,1495,1527],
[1559,1591,1623,1655,1687,1719,1751,1783],
[1815,1847,1879,1911,1943,1975,2007,2039],
[1048,1080,1112,1144,1176,1208,1240,1272],
[1304,1336,1368,1400,1432,1464,1496,1528],
[1560,1592,1624,1656,1688,1720,1752,1784],
[1816,1848,1880,1912,1944,1976,2008,2040],
[1049,1081,1113,1145,1177,1209,1241,1273],
[1305,1337,1369,1401,1433,1465,1497,1529],
[1561,1593,1625,1657,1689,1721,1753,1785],
[1817,1849,1881,1913,1945,1977,2009,2041],
[1050,1082,1114,1146,1178,1210,1242,1274],
[1306,1338,1370,1402,1434,1466,1498,1530],
[1562,1594,1626,1658,1690,1722,1754,1786],
[1818,1850,1882,1914,1946,1978,2010,2042],
[1051,1083,1115,1147,1179,1211,1243,1275],
[1307,1339,1371,1403,1435,1467,1499,1531],
[1563,1595,1627,1659,1691,1723,1755,1787],
[1819,1851,1883,1915,1947,1979,2011,2043],
[1052,1084,1116,1148,1180,1212,1244,1276],
[1308,1340,1372,1404,1436,1468,1500,1532],
[1564,1596,1628,1660,1692,1724,1756,1788],
[1820,1852,1884,1916,1948,1980,2012,2044],
[1053,1085,1117,1149,1181,1213,1245,1277],
[1309,1341,1373,1405,1437,1469,1501,1533],
[1565,1597,1629,1661,1693,1725,1757,1789],
[1821,1853,1885,1917,1949,1981,2013,2045],
[1054,1086,1118,1150,1182,1214,1246,1278],
[1310,1342,1374,1406,1438,1470,1502,1534],
[1566,1598,1630,1662,1694,1726,1758,1790],
[1822,1854,1886,1918,1950,1982,2014,2046],
[1055,1087,1119,1151,1183,1215,1247,1279],
[1311,1343,1375,1407,1439,1471,1503,1535],
[1567,1599,1631,1663,1695,1727,1759,1791],
[1823,1855,1887,1919,1951,1983,2015,2047],
]

#orig
'''
test_cases=[]
#note 64 between the comm get from 8k python shell, convert to 128
for mesh in ep_comm:
    for x in mesh:
        for y in mesh:
            if x!=y and x//64 != y//64:
                source = x%64 + x//64*128
                dest = y%64 + y//64*128
                test_cases.append((source, dest, 31))
print("test case len", len(test_cases) )
'''

#small test
test_cases=[]
#limit_path_num=1
for mesh in ep_comm: #[0:4]:
    tmp=mesh[::2]
    print("ep_comm", mesh, mesh[::2], ' | ', tmp[0]//64, tmp[1]//64, tmp[2]//64, tmp[3]//64)
    for x in mesh[::2]:
        for y in mesh[::2]:
            if x!=y and x//64 != y//64:
                source = x%64 + x//64*128
                dest = y%64 + y//64*128
                test_cases.append((source, dest, 1))
                #print('| ' , source//128, dest//128)
print("test case len", len(test_cases) )

#exit(0) 

#just for verify
n=0
for test in test_cases:
    if test[0]//128 == 0 and test[1]//128 == 1:
        #print(test[0], test[1])
        n+=1
print('0-1', n)

'''
test_cases=[
(1,128,31),
(1,256,31),
(1,384,31),
#(1025,128,61), 
#(1089,128,31), 
]
'''

path_log={}


#multiprocess the test_cases
num_threads=12
limit_path_num=15
limit_jump=8
with multiprocessing.Pool(processes = num_threads) as pool:
    input_arg_map=[]
    for test in test_cases:#[0:16]:
        arg=(G2, test[0], test[1], limit_path_num, "weight", dijkstra_with_builtin_heap)
        input_arg_map.append(arg)

    print(input_arg_map)
    results = pool.starmap(yenksp, input_arg_map)
    print(results)

#parse result
inter=0
for ret in results:
    num=0
    path_log[str(test_cases[inter][0])+'_'+str(test_cases[inter][1])]=[]
    for path in ret:
        remove=0
        length1 = len(path)
        if length1 >6:
            for n in range(length1):
                if n < length1 - 1:
                    sn=path[n]//128
                    sn1=path[n+1]//128
                    if sn != sn1 and sn//4 == sn1//4: # and length1 <=6:
                        # TODO: just print here now
                        remove=1
                        break
                        #print('n', n, sn, sn1, sn//4, sn1//4)

        # length limit
        if remove ==0 and length1 <= limit_jump:
            num+=1
            path_log[str(test_cases[inter][0])+'_'+str(test_cases[inter][1])].append(path)
        else:
            print("remove", end= "" )

        print(path, 'len=', len(path), "|", end="")
        for n in path:
            print(n//128, ',',end="")
        print()
    print("===================== get ", num, 'remove', test[2]-num)
    inter+=1



'''
#legacy single process mode
iteraton=0
for test in test_cases:
    iteraton+=1
    print("iteration -=============************************************>", iteraton)
    path_log[str(test[0])+'_'+str(test[1])]=[]
    ret=yenksp(G2, test[0], test[1], test[2], shortest_path_func=dijkstra_with_builtin_heap)
        
    num=0
    for path in ret:
        remove=0
        length1 = len(path)
        if length1 >6:
            for n in range(length1):
                if n < length1 - 1:
                    sn=path[n]//128
                    sn1=path[n+1]//128
                    if sn != sn1 and sn//4 == sn1//4: # and length1 <=6:
                        # TODO: just print here now
                        remove=1
                        break
                        #print('n', n, sn, sn1, sn//4, sn1//4)

        # length limit
        if remove ==0 and length1 <=8:
            num+=1
            path_log[str(test[0])+'_'+str(test[1])].append(path)
        else:
            print("remove", end= "" )

        print(path, 'len=', len(path), "|", end="")
        for n in path:
            print(n//128, ',',end="")
        print()
    print("===================== get ", num, 'remove', test[2]-num)

#print(path_log)
'''

#get link pressure
link_pressure={}
for sd in path_log:
    for link in path_log[sd]:
        length1=len(link)
        for n in range(length1-1):
            src=link[n]
            dst=link[n+1]
            key=str(src)+'_'+str(dst)
            if key in link_pressure:
                link_pressure[key] +=1
            else:
                link_pressure[key] =1

print("link pressure", len(link_pressure), link_pressure)


path_pressure={}
for sd in path_log:
    path_pressure[sd]=[]
    for link in path_log[sd]:
        pressure=[]
        length1=len(link)
        for n in range(length1-1):
            src=link[n]
            dst=link[n+1]
            key=str(src)+'_'+str(dst)
            pressure.append(link_pressure[key])

        path_pressure[sd].append(pressure)


print("path_pressure", len(path_pressure), path_pressure)

for sd in path_pressure:
    print("*", sd, len(path_log[sd]))
    for path in path_log[sd]:
        print(len(path), path, end=' ')
        for link in path:
            print( link//128, end=' ')
        print()
    for path in path_pressure[sd]:
        print("    ",path)
