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
        print('e=', e)
        e_weight[e] = edges[e][weight]
        if e[0] not in adj:
            adj[e[0]] = []
        adj[e[0]].append(e[1])
        print(adj)
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
num_union_node =8

num_node= num_D_node + num_union_node
d_row = 8
d_columns = 8

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
for l in range(0, d_row):
    union =  num_D_node + l
    for x in range(0+l, d_columns*d_row+l, d_row):
        G.add_edge(x, union, weight=1)
        G.add_edge(union, x, weight=1)
        for y in range(0+l, d_columns*d_row+l, d_row):
            if x!=y:
                G.add_edge(x, y, weight=1)

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

#for e in G.edges():
#    print(e)

ret=yenksp(G, 1, 9, 10, shortest_path_func=dijkstra_with_builtin_heap)
#ret=yenksp(G, 1, 9, 10)
print(ret)
