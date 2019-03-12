import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Node_Link_Path import Node, Link, Path
from misc import draw_network, get_literal, getKey
from exception import CantReach

class Graph:
    def __init__(self, file = 'first.txt', st = None):
        self.file = file
        self.st = st
        if st is None:
            self.nodes_list, self.links_list = self.load_graph_file()
        else:
            self.nodes_list, self.links_list = self.load_graph_st(st)

    def _data_for_bellman_ford(self, start):
        destination_dict = {} # Stands for destination
        pre_dict = {} # Stands for predecessor
        graph = self.nodes_to_dict()

        for node in graph:
            destination_dict[node] = float('Inf') # set inf distances to other node except start node
            pre_dict[node] = None
        destination_dict[start] = 0 # first node distance is zero couse it start
        return destination_dict, pre_dict, graph

    def relax(self, node_id, neigh_id, graph, d, p, path_list):
        '''
            if distance between node_id and neigh_id lower than we have
            remember this distance and node
        '''
        if d[neigh_id] > d[node_id] + graph[node_id][neigh_id]:
            # Record this lower distance
            d[neigh_id]  = d[node_id] + graph[node_id][neigh_id]
            p[neigh_id] = node_id
            #print(node_id)
            path_list[neigh_id].append(node_id)

        return d, p, path_list

    def bf_path(self, start, end=None) -> Path:
        d, p = self.bellman_ford(start);
        if end is not None:
            graph = self.nodes_to_dict()
            cur = p[end]
            path = [end]+[cur]
            for k in range(len(graph)-1):
                cur = p[cur]
                if cur is None:
                    return self.create_path(path[::-1])
                path.append(cur)
        else:
            return d


    def bellman_ford(self, start):
        '''
            find a shortest path from start_node to other node in graph
        '''
        d, p, graph = self._data_for_bellman_ford(start)
        path_list = dict((i, []) for i in graph.keys())

        for i in range(len(graph)-1):
            for u in graph:
                for v in graph[u]:
                    d, p, path_list = self.relax(u, v, graph, d, p, path_list)

        for u in graph:
            for v in graph[u]:
                if d[v] <= d[u] + graph[u][v]:
                    pass
                    #raise Exception("negative")
                    #print('negative')

        return d, p

    def find_all_paths(self, start, end, path = []) -> list:
        graph = self.nodes_to_dict()
        path = path + [start]
        if start == end:
            return [path]
        if not start in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self.find_all_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def bfs(self, start) -> list:
        '''
            returning all nodes from graph with bfs algo
            start - first node where to start

            FIFO structure
        '''
        graph = self.nodes_for_dbfs()
        visited, queue = set(), [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(graph[vertex] - visited)
        return visited

    def bfs_paths(self, start, end) -> list:
        '''
            finding all possible path with bfs algo
            start - where to start
            end - where to end

            FIFO structure
        '''
        graph = self.nodes_for_dbfs()
        queue = [(start, [start])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next in graph[vertex] - set(path):
                if next == end:
                    yield path + [next]
                else:
                    queue.append((next, path + [next]))


    def dfs(self, start, visited = None, reverse_order = None) -> list:
        '''
            simply returning all nodes from graph with dfs algorithm
            start - first node where to start
            visited - set with all visited nodes

            LIFO structure
        '''
        graph = self.nodes_for_dbfs()
        if visited is None:
            visited = set()
        visited.add(start)

        for next in graph[start] - visited:
            self.dfs(next, visited = visited, reverse_order = reverse_order)
        if isinstance(reverse_order, list):
            reverse_order.append(start)
        return visited

    def topological_sort(self) -> list:
        '''
            topological sort using recursive DFS

            graph = [
                [1, 6, 2],
                [1, 2, 2],
                [1, 3, 2],
                [2, 5, 3],
                [4, 6, 3],
                [4, 7, 2],
                [4, 5, 1],
                [4, 3, 1],
                [6, 3, 1],
                [7, 1, 1],
                [7, 5, 1],
            ]

            return: [5, 2, 3, 6, 1, 7, 4]
        '''
        visited = set()
        reversePostOrder = []
        graph = self.nodes_for_dbfs()
        for node in graph.keys():
            if node not in visited:
                self.dfs(node, visited = visited, reverse_order = reversePostOrder)
        return reversePostOrder

    def dfs_paths(self, start, end) -> list:
        '''
            finding all possible paths from start_node to end_node
            with dfs algorithm
            start - where to start node
            end - where to end node

            LIFO structure
        '''
        graph = self.nodes_for_dbfs()
        stack = [(start, [start])]
        while stack:
            (vertex, path) = stack.pop()
            for next in graph[vertex] - set(path):
                if next == end:
                    yield path + [next]
                else:
                    stack.append((next, path + [next]))

        return stack

    def fw_path(self, start = None, end = None) -> Path:
        d, p = self.floyd_warshall();
        if end is not None and start is not None:
            '''return path based on floyd_warshall algo'''
            need_node = p[start]
            cur = need_node[end]
            path = [end]+[cur]
            for k in range(len(need_node)-1):
                cur = need_node[cur]
                if cur==-1:
                    return self.create_path(path[::-1])
                path.append(cur)
        else:
            '''return distance matrix from all to all nodes'''
            return d

    def floyd_warshall(self):
        graph = self.nodes_to_dict()
        dist = {}
        pred = {}
        for u in graph:
            dist[u] = {}
            pred[u] = {}
            for v in graph:
                dist[u][v] = 1000
                pred[u][v] = -1
            dist[u][u] = 0
            for neighbor in graph[u]:
                dist[u][neighbor] = graph[u][neighbor]
                pred[u][neighbor] = u

        for t in graph:
            for u in graph:
                for v in graph:
                    newdist = dist[u][t] + dist[t][v]
                    if newdist < dist[u][v]:
                        dist[u][v] = newdist
                        pred[u][v] = pred[t][v] # route new path through t

        return dist, pred

    def nodes_to_dict(self) -> dict:
        '''
            returning oriented graph in representation which need for
            better performance in algo

            graph = {
              1: {1: 4},
              2: {1: 4, 3: 6, 4: 3, 5: 6},
              3: {2: 6, 5: 4},
              4: {2: 3, 5: 2},
              5: {2: 6, 3: 4, 4:2, 6: 5},
              6: {5: 5}
            }
            where key is node and value - weight
        '''
        graph = {}
        for key, node in self.get_all_nodes().items():
            links = graph.get(key, {})
            for link in node.get_linked_to():
                links.update({link.to_id:link.weight})
                graph.update({key: links})
            graph.update({key: links})

        return graph

    def nodes_for_dbfs(self) -> dict:
        '''
            smth like nodes_for_dejkstra but without weight
            graph = {
              1: {1},
              2: {1, 3, 4, 5},
              3: {2, 5},
              4: {2, 5},
              5: {2, 3, 4, 6},
              6: {5}
            }
        '''
        g = self.nodes_to_dict()
        for key, nodes in g.items():
            g.update({key:set([i for i in nodes.keys()])})
        return g

    def dejkstra_path(self, start, end = None) -> Path:
        opened_node = {} #p #словарь {открытая вершина : её метка}
        short_path = {} #b #словарь для отслеживания короткого пути
        visited_nodes = [] #t #список посещённых вершин
        current_node = start #v #текущая вершина
        end_node = end #e #конечная вершина
        opened_node[current_node] = 0
        short_path[current_node] = -1
        graph = self.nodes_to_dict()

        return self.dejkstra(graph, current_node, opened_node,
                            visited_nodes, short_path, end_node = end_node)

    def dejkstra(self, graph, current_node, opened_node, visited_nodes, short_path, end_node = None) -> list:
        node_count = len(graph)

        for x in graph[current_node]: #для каждого соседа (х) текущей вершины (current_node)
            xm = opened_node[current_node] + graph[current_node][x] #новая метка соседа (xm) =
                                #метка текущей вершины (p[current_node]) +
                                #значение ребра current_nodex (graph[current_node][x])

            if not x in opened_node: #если соседа (x) нет в словаре (opened_node)
                opened_node[x] = xm #записываем новую метку (xm) в словарь с ключем (x)
                short_path[x] = current_node  #как только метка пересчитывается, запоминаем
                          #(следующая вершина: предыдущая вершина) в словаре (short_path)
            elif not x in visited_nodes: #иначе если (x) не в (t)
                if opened_node[x] > xm: #если старая метка соседа больше новой метки
                    opened_node[x] = xm #новую метку записываем на место старой
                    short_path[x] = current_node #как только метка пересчитывается, запоминаем
                             #(следующая вершина: предыдущая вершина) в словаре (short_path)

            #print('текущей вершины current_node =', current_node, ' сосед x =', x, 'c меткой xm =', xm)

        #print('opened_node =', opened_node)

        #print('\n  Добавляем текущую вершину в список посещенных')
        visited_nodes.append(current_node)
        #print('visited_nodes =', visited_nodes)

        if node_count <= len(visited_nodes): # Условие выхода из функции
            s = [] #кратчайший путь
            s.insert(0, end_node) #вставляем (е) в список (s) по индексу (0)

            if end_node is not None:
                while True:

                    es = short_path.get(end_node, None)
                    if es == -1: #значение ключа (-1) имеет начальная вершина
                                   #вот её и ищем в словаре (short_path)
                        return self.create_path(s)
                    elif es is None:
                        raise CantReach("Cant reach {}".format(end_node))
                    end_node = short_path[end_node] #теперь последней вершиной будет предыдущая
                    s.insert(0, end_node) #вставляем (е) в список (s) по индексу (0)

                return s
            else:
                return short_path
        dm = opened_node[visited_nodes[0]]
        #print('\n  Находим вершину с минимальной меткой за исключением тех, что уже в visited_nodes')
        for d in opened_node: #вершина (d) с минимальной меткой из словаря (opened_node)
            if d not in visited_nodes:
                dm = opened_node[d] #метка вершины (d)
                break #пусть это будет первая вершина из словаря (opened_node)

        for y in opened_node: #для каждой вершины (y) из словаря (opened_node)
            if opened_node[y] < dm and not y in visited_nodes: #если метка вершины (y) <
                                         #метки вершины (d) & (y) нет в (t)
                dm = opened_node[y] #метку вершины (y) записываем в (dm)
                d = y #вершину (y) записываем в (d)
                #print('Вершина y =', y, 'с меткой dm =', dm)

        #print('Вершина d =', d, 'имеет минимальную метку dm =', dm, \
              #'\nтеперь текущей вершиной current_node будет вершина d')
        current_node = d #теперь текущей вершиной current_node будет вершина d
        return self.dejkstra(graph, current_node, opened_node,
                            visited_nodes, short_path, end_node)


    def create_path(self, path) -> Path:
        return Path(self, path)

    def update_node(self, temp_storage, node_id, link_to, weight) -> Node:
        node = temp_storage['nodes'].get(node_id)
        to_node = temp_storage['nodes'].get(link_to)
        if to_node is None:
            to_node = Node(link_to, self)
            temp_storage['nodes'].update({link_to: to_node})

        if node is None:
            node = Node(node_id, self)

        link = node.add_link(link_to, weight, self)

        temp_storage['links'].append(link)
        temp_storage['nodes'].update({node_id: node})
        return node

    def _node_count(self, temp_st, st):
        for i in st:
            temp_st['nodes'].update({int(i[0]): Node(i[0], self)})

        return temp_st

    def load_graph_st(self, st) -> dict:
        t = {'nodes':{}, 'links':[]}
        t = self._node_count(t, st)
        print(t)
        for line in st:
            id, to_id, w = line[0], line[1], line[2]
            node = self.update_node(t, id, to_id, w)

        return t['nodes'], t['links']

    def load_graph_file(self) -> dict:
        t = {'nodes':{}, 'links':[]}
        l = []
        for line in open(self.file, 'r', encoding = 'utf8').readlines():
            line = line.replace('\n', '').split(' ')
            id, to_id, w = int(line[0]), int(line[1]), int(line[2])
            l.append([id, to_id, w])
        t = self._node_count(t, l)
        for i in l:
            self.update_node(t, i[0], i[1], i[2])

        return t['nodes'], t['links']

    def nodes_id_to_links(self, path) -> list:
        return [self.get_link(path[i], path[i+1]) for i in range(0, len(path), 1) if i<len(path)-1]

    def get_link(self, from_id, to_id) -> Link:
        for link in self.links_list:
            if link.from_id == from_id and link.to_id == to_id:
                return link

    def get_links(self) -> list:
        return self.links_list

    def get_nodes(self) -> list:
        return self.nodes_list.values()

    def get_all_nodes(self) -> dict:
        return self.nodes_list

    def matrix_incident(self) -> np.matrix:
        matrix = np.zeros(shape = (len(self.nodes_list), len(self.links_list)))

        for n, i in enumerate(self.links_list):
            matrix[i.from_id-1, n] = -1
            matrix[i.to_id-1, n] = 1

        return matrix

    def matrix_adjacency(self) -> np.matrix:
        matrix = np.zeros(shape = (len(self.nodes_list), len(self.nodes_list)))

        for node in self.nodes_list.values():
            for link in node.get_linked_to():
                matrix[link.from_id-1, link.to_id-1] = 1

        return matrix

    def get_nodes_power(self) -> dict:
        l = {}
        for node in self.get_nodes():
            for link in node.get_linked_to():
                l.update({link.from_id: l.get(link.from_id, 0)+1})
                l.update({link.to_id: l.get(link.to_id, 0)+1})

        return l

    def get_isolated(self) -> list:
        nodes = self.get_nodes()
        l = dict((num, 0) for num in range(1, len(nodes)+1, 1))
        for node in nodes:
            for link in node.get_linked_to():
                if link.from_id != link.to_id:
                    l.update({link.from_id: l.get(link.from_id, 0)+1})

        return [i for i in l.keys() if l[i]==0]

    def show(self, path = None, save_file = None, show = True):
        G=nx.MultiDiGraph()
        fig, ax = plt.subplots()
        nodes = self.get_nodes()
        links = self.get_links()
        labels = {}
        if path:
            if not isinstance(path[0], Link):
                path = self.nodes_id_to_links(path)
        for n, node in enumerate(nodes):
            G.add_node(node.id)
            labels[n+1] = str(node.id)#get_literal(node.id)#str(node.id)
        for link in links:
            G.add_edge(link.from_id, link.to_id, weight = link.weight)

        pos=nx.circular_layout(G, center = [0,0])
        nx.draw_networkx_labels(G, pos, labels ,font_size=11)
        bbox_props = dict(boxstyle="round,pad=0.3", ec="k", lw=2)
        nx.draw_networkx_edge_labels(G,pos,bbox = bbox_props, label_pos = 0.6, font_size = 8, alpha = 1, edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)})
        ax=plt.gca()
        draw_network(G, pos, ax, links, path_links = path)
        ax.autoscale()
        #nx.draw(G, pos, with_labels=True)
        if save_file is not None:
            print('saving')
            plt.savefig(save_file, dpi=200)
        if show:
            plt.show()

    def __str__(self):
        representation = "Graph: \n"
        for node in self.get_nodes():
            representation += " "*4+str(node)+"\n"
            for link in node.get_linked_to():
                representation += " "*8+str(link)+"\n"

        return representation
