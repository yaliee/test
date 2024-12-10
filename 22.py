graph = {'0': set(['1', '2', '3', '4', '5', '6', '7']),
         '1': set(['0', '2', '3', '4', '5', '6', '7']),
         '2': set(['1', '0', '3', '4', '5', '6', '7']),
         '3': set(['1', '2', '0', '4', '5', '6', '7']),
         '4': set(['1', '2', '3', '0', '5', '6', '7']),
         '5': set(['1', '2', '3', '4', '0', '6', '7']),
         '6': set(['1', '2', '3', '4', '5', '0', '7']),
         '7': set(['1', '2', '3', '4', '5', '6', '0']),
        }

graph_small = {'0': set(['1', '2', '3', ]),
         '1': set(['0', '2', '3', ]),
         '2': set(['1', '0', '3', ]),
         '3': set(['1', '2', '0', ]),
        }

graph_with_1union_16 = {
         '0': set(['1', '2', '3', '4', '5', '6', '7', 'u1', '8']),
         '1': set(['0', '2', '3', '4', '5', '6', '7', 'u1', '9']),
         '2': set(['1', '0', '3', '4', '5', '6', '7', 'u1', '10']),
         '3': set(['1', '2', '0', '4', '5', '6', '7', 'u1', '11']),
         '4': set(['1', '2', '3', '0', '5', '6', '7', 'u1', '12']),
         '5': set(['1', '2', '3', '4', '0', '6', '7', 'u1', '13']),
         '6': set(['1', '2', '3', '4', '5', '0', '7', 'u1', '14']),
         '7': set(['1', '2', '3', '4', '5', '6', '0', 'u1', '15']),
         'u1': set(['1', '2', '3', '4', '5', '6', '0', '7']),

         '8': set(['9', '10', '11', '12', '13', '14', '15', 'u2', '0']),
         '9': set(['8', '10', '11', '12', '13', '14', '15', 'u2', '1']),
         '10': set(['9', '8', '11', '12', '13', '14', '15', 'u2', '2']),
         '11': set(['9', '10', '8', '12', '13', '14', '15', 'u2', '3']),
         '12': set(['9', '10', '11', '8', '13', '14', '15', 'u2', '4']),
         '13': set(['9', '10', '11', '12', '8', '14', '15', 'u2', '5']),
         '14': set(['9', '10', '11', '12', '13', '8', '15', 'u2', '6']),
         '15': set(['9', '10', '11', '12', '13', '14', '8', 'u2', '7']),
         'u2': set(['9', '10', '11', '12', '13', '14', '8', '15']),
        }

graph_with_1union_24 = {
         '0': set(['1', '2', '3', '4', '5', '6', '7', 'u1', '8', '16']),
         '1': set(['0', '2', '3', '4', '5', '6', '7', 'u1', '9', '17']),
         '2': set(['1', '0', '3', '4', '5', '6', '7', 'u1', '10', '18']),
         '3': set(['1', '2', '0', '4', '5', '6', '7', 'u1', '11', '19']),
         '4': set(['1', '2', '3', '0', '5', '6', '7', 'u1', '12', '20']),
         '5': set(['1', '2', '3', '4', '0', '6', '7', 'u1', '13', '21']),
         '6': set(['1', '2', '3', '4', '5', '0', '7', 'u1', '14', '22']),
         '7': set(['1', '2', '3', '4', '5', '6', '0', 'u1', '15', '23']),
         'u1': set(['1', '2', '3', '4', '5', '6', '0', '7']),

         '8': set(['9', '10', '11', '12', '13', '14', '15', 'u2', '0', '16']),
         '9': set(['8', '10', '11', '12', '13', '14', '15', 'u2', '1', '17']),
         '10': set(['9', '8', '11', '12', '13', '14', '15', 'u2', '2', '18']),
         '11': set(['9', '10', '8', '12', '13', '14', '15', 'u2', '3', '19']),
         '12': set(['9', '10', '11', '8', '13', '14', '15', 'u2', '4', '20']),
         '13': set(['9', '10', '11', '12', '8', '14', '15', 'u2', '5', '21']),
         '14': set(['9', '10', '11', '12', '13', '8', '15', 'u2', '6', '22']),
         '15': set(['9', '10', '11', '12', '13', '14', '8', 'u2', '7', '23']),
         'u2': set(['9', '10', '11', '12', '13', '14', '8', '15']),

         '16': set(['17', '18', '19', '20', '21', '22', '23', 'u3', '0', '8']),
         '17': set(['16', '18', '19', '20', '21', '22', '23', 'u3', '1', '9']),
         '18': set(['17', '16', '19', '20', '21', '22', '23', 'u3', '2', '10']),
         '19': set(['17', '18', '16', '20', '21', '22', '23', 'u3', '3', '11']),
         '20': set(['17', '18', '19', '16', '21', '22', '23', 'u3', '4', '12']),
         '21': set(['17', '18', '19', '20', '16', '22', '23', 'u3', '5', '13']),
         '22': set(['17', '18', '19', '20', '21', '16', '23', 'u3', '6', '14']),
         '23': set(['17', '18', '19', '20', '21', '22', '16', 'u3', '7', '15']),
         'u3': set(['17', '18', '19', '20', '21', '22', '16', '23']),
        }

graph_with_1union_32 = {
         '0': set(['1', '2', '3', '4', '5', '6', '7', 'u1', '8', '16']),
         '1': set(['0', '2', '3', '4', '5', '6', '7', 'u1', '9', '17']),
         '2': set(['1', '0', '3', '4', '5', '6', '7', 'u1', '10', '18']),
         '3': set(['1', '2', '0', '4', '5', '6', '7', 'u1', '11', '19']),
         '4': set(['1', '2', '3', '0', '5', '6', '7', 'u1', '12', '20']),
         '5': set(['1', '2', '3', '4', '0', '6', '7', 'u1', '13', '21']),
         '6': set(['1', '2', '3', '4', '5', '0', '7', 'u1', '14', '22']),
         '7': set(['1', '2', '3', '4', '5', '6', '0', 'u1', '15', '23']),
         'u1': set(['1', '2', '3', '4', '5', '6', '0', '7']),

         '8': set(['9', '10', '11', '12', '13', '14', '15', 'u2', '0', '16']),
         '9': set(['8', '10', '11', '12', '13', '14', '15', 'u2', '1', '17']),
         '10': set(['9', '8', '11', '12', '13', '14', '15', 'u2', '2', '18']),
         '11': set(['9', '10', '8', '12', '13', '14', '15', 'u2', '3', '19']),
         '12': set(['9', '10', '11', '8', '13', '14', '15', 'u2', '4', '20']),
         '13': set(['9', '10', '11', '12', '8', '14', '15', 'u2', '5', '21']),
         '14': set(['9', '10', '11', '12', '13', '8', '15', 'u2', '6', '22']),
         '15': set(['9', '10', '11', '12', '13', '14', '8', 'u2', '7', '23']),
         'u2': set(['9', '10', '11', '12', '13', '14', '8', '15']),

         '16': set(['17', '18', '19', '20', '21', '22', '23', 'u3', '0', '8']),
         '17': set(['16', '18', '19', '20', '21', '22', '23', 'u3', '1', '9']),
         '18': set(['17', '16', '19', '20', '21', '22', '23', 'u3', '2', '10']),
         '19': set(['17', '18', '16', '20', '21', '22', '23', 'u3', '3', '11']),
         '20': set(['17', '18', '19', '16', '21', '22', '23', 'u3', '4', '12']),
         '21': set(['17', '18', '19', '20', '16', '22', '23', 'u3', '5', '13']),
         '22': set(['17', '18', '19', '20', '21', '16', '23', 'u3', '6', '14']),
         '23': set(['17', '18', '19', '20', '21', '22', '16', 'u3', '7', '15']),
         'u3': set(['17', '18', '19', '20', '21', '22', '16', '23']),
        }

graph_with_2union_16 = {
         '0': set(['1', '2', '3', '4', '5', '6', '7', 'u1', '8', 'u3']),
         '1': set(['0', '2', '3', '4', '5', '6', '7', 'u1', '9', 'u3']),
         '2': set(['1', '0', '3', '4', '5', '6', '7', 'u1', '10', 'u3']),
         '3': set(['1', '2', '0', '4', '5', '6', '7', 'u1', '11', 'u3']),
         '4': set(['1', '2', '3', '0', '5', '6', '7', 'u1', '12', 'u3']),
         '5': set(['1', '2', '3', '4', '0', '6', '7', 'u1', '13', 'u3']),
         '6': set(['1', '2', '3', '4', '5', '0', '7', 'u1', '14', 'u3']),
         '7': set(['1', '2', '3', '4', '5', '6', '0', 'u1', '15', 'u3']),
         'u1': set(['1', '2', '3', '4', '5', '6', '0', '7']),
         'u3': set(['1', '2', '3', '4', '5', '6', '0', '7']),

         '8': set(['9', '10', '11', '12', '13', '14', '15', 'u2', '0', 'u4']),
         '9': set(['8', '10', '11', '12', '13', '14', '15', 'u2', '1', 'u4']),
         '10': set(['9', '8', '11', '12', '13', '14', '15', 'u2', '2', 'u4']),
         '11': set(['9', '10', '8', '12', '13', '14', '15', 'u2', '3', 'u4']),
         '12': set(['9', '10', '11', '8', '13', '14', '15', 'u2', '4', 'u4']),
         '13': set(['9', '10', '11', '12', '8', '14', '15', 'u2', '5', 'u4']),
         '14': set(['9', '10', '11', '12', '13', '8', '15', 'u2', '6', 'u4']),
         '15': set(['9', '10', '11', '12', '13', '14', '8', 'u2', '7', 'u4']),
         'u2': set(['9', '10', '11', '12', '13', '14', '8', '15', '8']),
         'u4': set(['9', '10', '11', '12', '13', '14', '8', '15', '8']),
        }

graph_with_1union = {
         '0': set(['1', '2', '3', '4', '5', '6', '7', 'u1']),
         '1': set(['0', '2', '3', '4', '5', '6', '7', 'u1']),
         '2': set(['1', '0', '3', '4', '5', '6', '7', 'u1']),
         '3': set(['1', '2', '0', '4', '5', '6', '7', 'u1']),
         '4': set(['1', '2', '3', '0', '5', '6', '7', 'u1']),
         '5': set(['1', '2', '3', '4', '0', '6', '7', 'u1']),
         '6': set(['1', '2', '3', '4', '5', '0', '7', 'u1']),
         '7': set(['1', '2', '3', '4', '5', '6', '0', 'u1']),
         'u1': set(['1', '2', '3', '4', '5', '6', '0', '7']),
        }

graph_with_2union = {'0': set(['1', '2', '3', '4', '5', '6', '7', 'u1', 'u2']),
         '1': set(['0', '2', '3', '4', '5', '6', '7', 'u1', 'u2']),
         '2': set(['1', '0', '3', '4', '5', '6', '7', 'u1', 'u2']),
         '3': set(['1', '2', '0', '4', '5', '6', '7', 'u1', 'u2']),
         '4': set(['1', '2', '3', '0', '5', '6', '7', 'u1', 'u2']),
         '5': set(['1', '2', '3', '4', '0', '6', '7', 'u1', 'u2']),
         '6': set(['1', '2', '3', '4', '5', '0', '7', 'u1', 'u2']),
         '7': set(['1', '2', '3', '4', '5', '6', '0', 'u1', 'u2']),
         'u1': set(['1', '2', '3', '4', '5', '6', '0', '7',]),
         'u2': set(['1', '2', '3', '4', '5', '6', '0', '7',]),
        }

graph_with_4union = {'0': set(['1', '2', '3', '4', '5', '6', '7', 'u1', 'u2', 'u3', 'u4']),
         '1': set(['0', '2', '3', '4', '5', '6', '7', 'u1', 'u2', 'u3', 'u4']),
         '2': set(['1', '0', '3', '4', '5', '6', '7', 'u1', 'u2', 'u3', 'u4']),
         '3': set(['1', '2', '0', '4', '5', '6', '7', 'u1', 'u2', 'u3', 'u4']),
         '4': set(['1', '2', '3', '0', '5', '6', '7', 'u1', 'u2', 'u3', 'u4']),
         '5': set(['1', '2', '3', '4', '0', '6', '7', 'u1', 'u2', 'u3', 'u4']),
         '6': set(['1', '2', '3', '4', '5', '0', '7', 'u1', 'u2', 'u3', 'u4']),
         '7': set(['1', '2', '3', '4', '5', '6', '0', 'u1', 'u2', 'u3', 'u4']),
         'u1': set(['1', '2', '3', '4', '5', '6', '0', '7',]),
         'u2': set(['1', '2', '3', '4', '5', '6', '0', '7',]),
         'u3': set(['1', '2', '3', '4', '5', '6', '0', '7',]),
         'u4': set(['1', '2', '3', '4', '5', '6', '0', '7',]),
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
        #print('stack ', stack)
        (vertex, path) = stack.pop()
        #print("pop", vertex, path)

        if len(path) > global_jump:
            continue
        for next in graph[vertex] - set(path):
            #print("next ", next, 'vertex', vertex, 'graph[vertex]', graph[vertex], 'path', path)
            if next == goal:
                if len(path) <= global_jump:
                    #print('get path', path + [next])
                    yield path + [next]
            else:
                stack.append((next, path + [next]))



def convert_list_to_key( ll):
    return str(ll[0]) + '_' + str(ll[1])

def convert_key_to_list( ll):
    return str(ll[0]) + '_' + str(ll[1])

#================================================

import argparse

parser = argparse.ArgumentParser("simple_example")
parser.add_argument("-j", dest='jump', help="n jump", type=int, default=2)
parser.add_argument("-w", dest='working', help="n working", type=int, default=2)
parser.add_argument("-u", dest='union', help="union", type=int, default=0)
parser.add_argument("-p", dest='tp', help="tp", type=int, default=4)

args = parser.parse_args()

global_jump=args.jump


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

if args.union == 4:
    test_graph=graph_with_4union
elif args.union == 2:
    test_graph=graph_with_2union
elif args.union == 1:
    test_graph=graph_with_1union
elif args.union == 0:
    test_graph=graph
elif args.union == 161:
    test_graph=graph_with_1union_16
elif args.union == 162:
    test_graph=graph_with_2union_16
elif args.union == 241:
    test_graph=graph_with_1union_24
else:
    print("error 2")
    exit(1)

all_path={}

if args.tp == 4:

    # small 1 working,  2 benifit: 7/5,  2_union benifit: 8/5 
    for x in range(4):
        for y in range(4):
            if x==y:
                continue
            print(x, y)
            tmp = list(dfs_paths(test_graph, str(x), str(y)))
            all_path[str(x)+'_'+str(y)] = tmp
            AA.append(tmp)

    # small 2 working,  2 benifit : 7/6, 2_union benifit: 8/6

    if args.working==2:
        for x in range(4, 8):
            for y in range(4, 8):
                if x==y:
                    continue
                print(x, y)
                #AA.append(list(dfs_paths(test_graph, str(x), str(y))) )
                tmp = list(dfs_paths(test_graph, str(x), str(y)))
                all_path[str(x)+'_'+ str(y)] = tmp
                AA.append(tmp)


    # filter
    for item in AA:
        for path in item[:]:
            if len(path) != 2:
                if path[0] in ['0', '1', '2', '3'] and path[1] in ['0', '1', '2', '3']:
                    print("remove", path)
                    item.remove(path)
                if path[0] in ['4', '5', '6', '7'] and path[1] in ['4', '5', '6', '7']:
                    print("remove", path)
                    item.remove(path)
elif args.tp == 8:
    for x in range(8):
        for y in range(8):
            if x==y:
                continue
            print(x, y)
            #AA.append(list(dfs_paths(test_graph, str(x), str(y))) )
            tmp = list(dfs_paths(test_graph, str(x), str(y)))
            all_path[str(x)+'_'+ str(y)] = tmp
            AA.append(tmp)

    if args.working==2:
        for x in range(8, 16):
            for y in range(8, 16):
                if x==y:
                    continue
                print(x, y)
                #AA.append(list(dfs_paths(test_graph, str(x), str(y))) )
                tmp = list(dfs_paths(test_graph, str(x), str(y)))
                all_path[str(x)+'_'+ str(y)] = tmp
                AA.append(tmp)

    if args.working==3:
        for x in range(16, 24):
            for y in range(16, 24):
                if x==y:
                    continue
                print(x, y)
                #AA.append(list(dfs_paths(test_graph, str(x), str(y))) )
                tmp = list(dfs_paths(test_graph, str(x), str(y)))
                all_path[str(x)+'_'+ str(y)] = tmp
                AA.append(tmp)

    # filter
    removed=[]
    for item in AA:
        for path in item[:]:
            if len(path) != 2:
                tmp4 = ['0', '1', '2', '3', '4', '5', '6', '7' ]
                tmp5 = ['8', '9', '10', '11', '12', '13', '14', '15' ]
                tmp6 = ['16', '17', '18', '19', '20', '21', '22', '23' ]
                if len(path) <=3:
                    #tmp = ['0', '1', '2', '3', '4', '5', '6', '7' ]
                    #tmp2 = ['8', '9', '10', '11', '12', '13', '14', '15' ]
                    n = 0
                    while n < len(path)-1: 
                        if (path[n] in tmp4 and path[n+1] in tmp4) or (path[n] in tmp5 and path[n+1] in tmp5) or (path[n] in tmp6 and path[n+1] in tmp6) :
                            print("remove0", path)
                            item.remove(path)
                            removed.append(path)
                            break
                        n=n+1
                if len(path) >3:
                    tmp = ['0', '1', '2', '3', '4', '5', '6', '7', 'u1', 'u3']
                    tmp2 = ['8', '9', '10', '11', '12', '13', '14', '15', 'u2', 'u4' ]
                    tmp3 = ['u1', 'u2' ,'u3', 'u4']

                    '''
                    n = 0
                    while n < len(path)-2: 
                        if (path[n] in tmp and path[n+1] in tmp and path[n+2] in tmp) or (path[n] in tmp2 and path[n+1] in tmp2 and path[n+2] in tmp2) :
                            print("remove1", path)
                            item.remove(path)
                            removed.append(path)
                            break
                        n=n+1
                    '''


                    if path[0] in tmp and path[1] in tmp and path not in removed:
                        print("remove2", path)
                        item.remove(path)
                        removed.append(path)


                    n = 0
                    while n < len(path)-1:
                        if ((path[n] in tmp4 and path[n+1] in tmp4) or (path[n] in tmp5 and path[n+1] in tmp5) or (path[n] in tmp6 and path[n+1] in tmp6) ) and path not in removed:
                            print("remove3", path)
                            item.remove(path)
                            removed.append(path)
                            break
                        n=n+1

                    found =0 
                    if len(path) >=5:
                        for xx in tmp3:
                            if xx in path:
                                found = found+1

                        if found != 1 and path not in removed:
                                print("remove4", path)
                                item.remove(path)
                                removed.append(path)

                    '''
                    '''
                    '''
                    tmp = ['u1', 'u2' ,'u3', 'u4']
                    if path[1] not in tmp:
                        print("remove", path)
                        item.remove(path)
                    '''
                #if len(path) ==4:
                #    print("remove", path)
                #    item.remove(path)
                #if len(path) >5:
                #    tmp = ['u1', 'u2' ,'u3', 'u4']
                #    if path[3] not in tmp:
                #        print("remove", path)
                #        item.remove(path)

else:
    print("error")
    exit(1)

#AA.append(list(dfs_paths(graph, '4', '5')) )
#AA.append(list(dfs_paths(graph, '5', '6')) )
#AA.append(list(dfs_paths(graph, '6', '7')) )
#AA.append(list(dfs_paths(graph, '7', '4')) )

print("111 all ring path", AA)

#print("123", all_path)
#exit(0)

for n in range(len(AA)):
    print(len(AA[n]))

#exit(0)
'''
for item in AA:
    for path in item[:]:
        if len(path) == 3 and (path[1] == 'u18' or path[1] == 'u19' ) and (path[0] != 'bak' and path[2] != 'bak' ):
            print("remove", path)
            item.remove(path)
'''

#print("222 all ring path ", AA)
#for n in range(len(AA)):
#    print(len(AA[n]))


#===============================Filter done 
FIN=[]

# 路径分解：
for item in AA:
    for path in item:
        '''
        #print([path[0], path[1]])
        #tmp=sorted([path[0], path[1]])
        tmp=[path[0], path[1]]
        FIN.append(tmp)
        if len(path) == 3:
            tmp=[path[1], path[2]]
            FIN.append(tmp)
        '''
        length = len(path)
        n = 0 
        while n< length-1:
            tmp=[path[n], path[n+1]]
            FIN.append(tmp)
            n=n+1
        

print(FIN, "FIN len",len(FIN))

all=0

#FIN2=copy.copy(FIN)

path_cap = {}

#各个路径压力
while len(FIN) > 0:
    before = len(FIN)
    #print("before ", before)
    item=FIN[0]
    #print("item111", item)
    for bb in FIN[:]:
        if bb == item:
            FIN.remove(bb)

    #if len(FIN) ==  1 and FIN[0] == item:
    #        FIN.remove(bb)
    after = len(FIN)

    #print("before after ", before ,after)
    all= all+before-after
    print(before-after)
    path_cap[convert_list_to_key(item)] = before-after

print("all", all)
print("path_cap ", path_cap)

for item in all_path:
    print(item, len(all_path[item]))
    this_cap=[]
    print(all_path[item])
    for ii in all_path[item]:
        #print("ii ", ii)

        tmp = ii
        length = len(tmp)
        #print("tmp ", tmp, length)
        n = 0
        cap=[]
        while n < length-1:
            cap.append(path_cap[str(tmp[n])+'_'+str(tmp[n+1])])
            n=n+1
        this_cap.append(cap)
    print(this_cap)

    sum = 0
    for aa in this_cap:
        max=0
        for bb in aa:
            if bb > max:
                max=bb
        sum = sum + 1/max 
    print("max=", max, 'path num=', len(all_path[item]))
    #print("ratio decrease = ", 1 - max/len(all_path[item])) 
    print("ratio decrease ", 1 - 1/sum) 
