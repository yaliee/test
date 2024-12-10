from typing import Any, DefaultDict, List
from networkx.classes.graph import Graph
from heapq import heappop, heappush
from collections import defaultdict
import networkx as nx

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
            length, path = shortest_path_func(adjascent, source, target, edge_weight)
            if isinstance(path, list):
                listB.push(length, path)
            else:
                raise NotImplementedError(f"No path exists between node {source} and node {target}.")
        else:
            ignore_nodes = set()

            #wyl add, skip Dnode in rack if comm between different racks
            ss=(source)//128
            tt=(target)//128
            #print("ss", ss, "tt", tt);
            if ss!=tt:
                D_range=range(ss*128,  ss*128+num_D_node)
                if ss in D_range:
                    tmp=G2.in_edges(source)
                    for xx in tmp:
                        if xx[0] in D_range and xx[1] in D_range:
                            for sl in xx:
                                if sl != source:
                                    ignore_nodes.add(sl)
            
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = sum([edge_weight[(u,v)] for u,v in zip(root, root[1:])])
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i-1],path[i]))
                length, supr = shortest_path_func(adjascent, root[-1], target, edge_weight, ignore_node=ignore_nodes, ignore_edge=ignore_edges)
                if isinstance(supr, list):
                    listB.push(root_length+length, root[:-1]+supr)
                ignore_nodes.add(root[-1])
        if listB:
            listA.append(listB.pop())
            prev_path = listA[-1]
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

test_num_list=[0,1,64,100, 96]
#for range_rack_n in range(0, num_all_node, num_per_rack_node):
for n in range(rack_num):
    current_rack_node_range=node_list[(n)*num_per_rack_node:(n+1)*num_per_rack_node]
    print("rack ", n, current_rack_node_range)

    # 横排
    for l in range(1, d_columns+1):
        for x in range(d_row*(l-1), d_row*l):
            for y in range(d_row*(l-1), d_row*l):
                if x!=y:
                   G2.add_edge(x+n*num_per_rack_node, y+n*num_per_rack_node, weight=1)

    for test_num in test_num_list:
        print("test1", test_num, G2.in_edges(test_num), len(G2.in_edges(test_num)))

    # 纵排Ｄ and L1 union
    for nr in range(0, rail):
        for l in range(0, d_columns):
            print(num_D_node, l, nr, d_columns)
            union =  num_D_node + l + nr*d_columns
            for x in range(0+l, d_columns*d_row+l, d_columns):
                # L2 union
                #print("conn", x+ n*num_per_rack_node, union+ n*num_per_rack_node, union, l, x)
                G2.add_edge(x+ n*num_per_rack_node, union+ n*num_per_rack_node, weight=1)
                G2.add_edge(union + n*num_per_rack_node , x + n*num_per_rack_node, weight=1)
                for y in range(0+l, d_columns*d_row+l, d_columns):
                    if x!=y:
                        G2.add_edge(x+ n*num_per_rack_node, y + n*num_per_rack_node, weight=1)
    for test_num in test_num_list:
        print("test1", test_num, G2.in_edges(test_num), len(G2.in_edges(test_num)))

    # L1 and L2 union
    for nr in range(0, rail):
        for x in range(num_D_node + nr*d_columns, num_D_node+d_columns*(nr+1)):
            for y in range(num_D_node+ num_l1_union_node + nr*d_columns, num_D_node+ num_l1_union_node + d_columns*(nr+1)):
                #print("conn", x, y)
                G2.add_edge(x + n*num_per_rack_node, y + n*num_per_rack_node, weight=1)
                G2.add_edge(y + n*num_per_rack_node, x + n*num_per_rack_node, weight=1)

    for test_num in test_num_list:
        print("test1", test_num, G2.in_edges(test_num), len(G2.in_edges(test_num)))

    # 2D between rack
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

            print("conn", (xx)//128, xx%128, (int)(yy)//128, yy%128, index1, index2)
            index1 = (index1+1)% num_l2_union_node
            index2 = (index2)% num_l2_union_node

    # 1.5 D

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
ret=yenksp(G2, 2, 1000, 11, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
ret=yenksp(G2, 0, 128, 11, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
print("=====================")
ret=yenksp(G2, 0, 1, 11, shortest_path_func=dijkstra_with_builtin_heap)
print(ret)
