graph = {'0': set(['1', '2', '3', '4', '5', '6', '7']),
         '1': set(['0', '2', '3', '4', '5', '6', '7']),
         '2': set(['1', '0', '3', '4', '5', '6', '7']),
         '3': set(['1', '2', '0', '4', '5', '6', '7']),
         '4': set(['1', '2', '3', '0', '5', '6', '7']),
         '5': set(['1', '2', '3', '4', '0', '6', '7']),
         '6': set(['1', '2', '3', '4', '5', '0', '7']),
         '7': set(['1', '2', '3', '4', '5', '6', '0']),
        }

graph_fault = {
         '0': set(['1', '2', '3', '5', '6', '7',  'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']),
         '1': set(['0', '2', '3', '5', '6', '7', 'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']),
         '2': set(['1', '0', '3', '5', '6', '7', 'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']),
         '3': set(['1', '2', '0', '5', '6', '7', 'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']),
# fault  '4': set(['1', '2', '3', '0', '5', '6', '7']),
         'bak': set([ 'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']), 
         '5': set(['1', '2', '3', '0', '6', '7', 'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']),
         '6': set(['1', '2', '3', '5', '0', '7', 'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']),
         '7': set(['1', '2', '3', '5', '6', '0', 'u18', 'u19' ]), #'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17']),
         'u18': set(['1', '2', '3', '5', '6', '0', '7', 'bak']), 
         'u19': set(['1', '2', '3', '5', '6', '0', '7', 'bak']), 

        }

def dfs_paths(graph, start, goal): # iterative
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                if len(path) <=4:
                    yield path + [next]
            else:
                stack.append((next, path + [next]))

AA=[]

'''
AA.append(list(dfs_paths(graph_fault, '0', '1')) )
AA.append(list(dfs_paths(graph_fault, '1', '2')) )
AA.append(list(dfs_paths(graph_fault, '2', '3')) )
AA.append(list(dfs_paths(graph_fault, '3', '0')) )

AA.append(list(dfs_paths(graph_fault, 'bak', '5')) )
AA.append(list(dfs_paths(graph_fault, '5', '6')) )
AA.append(list(dfs_paths(graph_fault, '6', '7')) )
AA.append(list(dfs_paths(graph_fault, '7', 'bak')) )

'''
AA.append(list(dfs_paths(graph, '0', '1')) )
AA.append(list(dfs_paths(graph, '1', '2')) )
AA.append(list(dfs_paths(graph, '2', '3')) )
AA.append(list(dfs_paths(graph, '3', '0')) )

AA.append(list(dfs_paths(graph, '4', '5')) )
AA.append(list(dfs_paths(graph, '5', '6')) )
AA.append(list(dfs_paths(graph, '6', '7')) )
AA.append(list(dfs_paths(graph, '7', '4')) )


print("111 all ring path ", AA)
for n in range(len(AA)):
    print(len(AA[n]))

for item in AA:
    for path in item[:]:
        if len(path) == 3 and (path[1] == 'u18' or path[1] == 'u19' ) and (path[0] != 'bak' and path[2] != 'bak' ):
            print("remove", path)
            item.remove(path)

print("222 all ring path ", AA)
for n in range(len(AA)):
    print(len(AA[n]))

FIN=[]

# 路径分解：
for item in AA:
    for path in item:
        #print([path[0], path[1]])
        #tmp=sorted([path[0], path[1]])
        tmp=[path[0], path[1]]
        FIN.append(tmp)
        if len(path) == 3:
            #print([path[1], path[2]])
            #tmp=sorted([path[1], path[2]])
            tmp=[path[1], path[2]]
            FIN.append(tmp)

print(FIN, "FIN len",len(FIN))

all=0

#FIN2=copy.copy(FIN)

while len(FIN) > 0:
    before = len(FIN)
    #print("before ", before)
    item=FIN[0]
    print(item)
    for bb in FIN[:]:
        if bb == item:
            FIN.remove(bb)

    #if len(FIN) ==  1 and FIN[0] == item:
    #        FIN.remove(bb)
    after = len(FIN)

    #print("before after ", before ,after)
    all= all+before-after
    print(before-after)

print("all", all)
