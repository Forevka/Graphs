import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from misc import draw_network, get_literal, getKey

class Link:
    def __init__(self, from_id, to_id, weight, father):
        self.from_id = int(from_id)
        self.to_id = int(to_id)
        self.weight = int(weight)
        self.father_graph = father

    def get_node_from(self):
        nodes = self.father_graph.get_all_nodes()
        return nodes[self.from_id]

    def get_node_to(self):
        nodes = self.father_graph.get_all_nodes()
        return nodes[self.to_id]

    def get_weight(self):
        return self.weight

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Link from {} to {} weight {}".format(self.from_id, self.to_id, self.weight)

class Node:
    def __init__(self, id, father):
        self.id = int(id)
        self.father_graph = father
        self.links = []
        self.marked_links = []

    def unmark_links(self):
        self.marked_links = []

    def mark_link(self, link):
        self.marked_links.append(link)

    def get_linked_from(self) -> Link:
        all_links = father.get_links()
        for link in all_links:
            if link.from_id == self.id:
                yield link

    def get_linked_to(self) -> Link:# pass
        for i in self.links:
            yield i

    def get_cheapest_link(self, offset) -> Link:
        all_links = self.links
        s = sorted([link for link in all_links if link not in self.marked_links], key=getKey)
        print(s)
        return s[offset]

    def add_link(self, to_id, weight, father) -> Link:
        new_link = Link(self.id, to_id, weight, father)
        self.links.append(new_link)
        return new_link

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Node {} Links count: {}".format(self.id, len(self.links))



class Graph:
    def __init__(self, file = 'first.txt', st = None):
        self.file = file
        if st is None:
            self.nodes_list, self.links_list = self.load_graph_file()
        else:
            self.nodes_list, self.links_list = self.load_graph_st(st)
        print(self.nodes_list)

    def find_all_paths(self, start, end, path = []):
        graph = self.nodes_for_dejkstra()
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

    def nodes_for_dejkstra(self) -> dict:
        graph = {}
        for key, node in self.get_all_nodes().items():
            for link in node.get_linked_to():
                links = graph.get(key, {})
                links.update({link.to_id:link.weight})
                graph.update({key: links})
            graph.update({key: links})

        return graph

    def start_dejkstra(self, start, end):
        opened_node = {} #p #словарь {открытая вершина : её метка}
        short_path = {} #b #словарь для отслеживания короткого пути
        visited_nodes = [] #t #список посещённых вершин
        current_node = start #v #текущая вершина
        end_node = end #e #конечная вершина
        opened_node[current_node] = 0
        short_path[current_node] = -1
        #print('\n  Начальная текущая вершина v =', v)
        graph = self.nodes_for_dejkstra()
        print(graph)
        return self.dejkstra(graph, current_node, opened_node,
                            visited_nodes, short_path, end_node)

    def dejkstra(self, graph, current_node, opened_node, visited_nodes, short_path, end_node):
        print('\n  Обходим всех соседей текущей вершины')
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

            print('текущей вершины current_node =', current_node, ' сосед x =', x, 'c меткой xm =', xm)

        print('opened_node =', opened_node)

        print('\n  Добавляем текущую вершину в список посещенных')
        visited_nodes.append(current_node)
        print('visited_nodes =', visited_nodes)

        if node_count <= len(visited_nodes): # Условие выхода из функции
            print('\nВсё!\nВершины и их метки =', opened_node)
            print('Словарь для отслеживания пути =', short_path)

            s = [] #кратчайший путь
            s.insert(0, end_node) #вставляем (е) в список (s) по индексу (0)

            while True:
                if short_path[end_node] == -1: #значение ключа (-1) имеет начальная вершина
                               #вот её и ищем в словаре (short_path)
                    print('Кратчайший путь от начальной до конечной вершины =', s)
                    return s
                end_node = short_path[end_node] #теперь последней вершиной будет предыдущая
                s.insert(0, end_node) #вставляем (е) в список (s) по индексу (0)

            return  s

        print('\n  Находим вершину с минимальной меткой за исключением тех, что уже в visited_nodes')
        for d in opened_node: #вершина (d) с минимальной меткой из словаря (opened_node)
            if d not in visited_nodes:
                dm = opened_node[d] #метка вершины (d)
                break #пусть это будет первая вершина из словаря (opened_node)

        for y in opened_node: #для каждой вершины (y) из словаря (opened_node)
            if opened_node[y] < dm and not y in visited_nodes: #если метка вершины (y) <
                                         #метки вершины (d) & (y) нет в (t)
                dm = opened_node[y] #метку вершины (y) записываем в (dm)
                d = y #вершину (y) записываем в (d)
                print('Вершина y =', y, 'с меткой dm =', dm)

        print('Вершина d =', d, 'имеет минимальную метку dm =', dm, \
              '\nтеперь текущей вершиной current_node будет вершина d')
        current_node = d #теперь текущей вершиной current_node будет вершина d

        print('\n  Рекурсивно вызываем функцию Дейкстры с параметрами {}, opened_node, visited_nodes, short_path, end_node'.format(current_node))
        return self.dejkstra(graph, current_node, opened_node,
                            visited_nodes, short_path, end_node)


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

    def get_link(self, from_id, to_id):
        for link in self.links_list:
            if link.from_id == from_id and link.to_id == to_id:
                return link

    def get_links(self) -> list:
        return self.links_list

    def get_nodes(self) -> list:
        return self.nodes_list.values()

    def get_all_nodes(self) -> dict:
        return self.nodes_list

    def matrix_incident(self):
        matrix = np.zeros(shape = (len(self.nodes_list), len(self.links_list)))

        for n, i in enumerate(self.links_list):
            matrix[i.from_id-1, n] = -1
            matrix[i.to_id-1, n] = 1

        return matrix

    def matrix_adjacency(self):
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

    def nodes_id_to_links(self, path):
        return [self.get_link(path[i], path[i+1]) for i in range(0, len(path), 1) if i<len(path)-1]

    def get_path_length(self, path):
        return sum([self.get_link(path[i], path[i+1]).get_weight() for i in range(0, len(path), 1) if i<len(path)-1])

    def get_isolated(self):
        nodes = self.get_nodes()
        l = dict((num, 0) for num in range(1, len(nodes)+1, 1))
        for node in nodes:
            for link in node.get_linked_to():
                if link.from_id != link.to_id:
                    l.update({link.from_id: l.get(link.from_id, 0)+1})

        return [i for i in l.keys() if l[i]==0]

    def my_find_path(self, from_id, to_id, stack = []):
        '''dont good code need to rewrite'''
        stack.append(from_id)
        this_node = self.get_all_nodes()[from_id]
        #print(this_node)
        finded = False
        offset = 0
        while(from_id != to_id):
            current_link = this_node.get_link()#get_cheapest_link(offset)
            #offset +=1
            this_node.mark_link(current_link)
            #print(current_link)
            new_node = current_link.get_node_to()

            if new_node.id!=to_id:
                while(new_node.id in self.get_isolated()):
                    #print('isolated')
                    #print(offset)
                    this_node = current_link.get_node_from()
                    #print(this_node)
                    current_link = this_node.get_link()#get_cheapest_link(offset-1)
                    new_node = current_link.get_node_to()
                    #print(new_node)
                    #offset+=1
            self.my_find_path(new_node.id, to_id, stack = stack)
            return stack

    def show(self, path = None, save_file = None):
        G=nx.MultiDiGraph()
        fig, ax = plt.subplots()
        nodes = self.get_nodes()
        links = self.get_links()
        labels = {}
        path_links = []
        if path is not None:
            path_links = self.nodes_id_to_links(path)
        for n, node in enumerate(nodes):
            G.add_node(node.id)
            labels[n+1] = str(node.id)#get_literal(node.id)#str(node.id)
        for link in links:
            G.add_edge(link.from_id, link.to_id, weight = link.weight)

        pos=nx.circular_layout(G)
        nx.draw_networkx_labels(G, pos, labels ,font_size=11)
        bbox_props = dict(boxstyle="round,pad=0.3", ec="k", lw=2)
        nx.draw_networkx_edge_labels(G,pos,bbox = bbox_props, label_pos = 0.6, font_size = 8, alpha = 1, edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)})
        ax=plt.gca()
        draw_network(G, pos, ax, links, path_links = path_links)
        ax.autoscale()
        #nx.draw(G, pos, with_labels=True)
        if save_file is not None:
            print('saving')
            plt.savefig(save_file)

        plt.show()

    def __str__(self):
        representation = "Graph with:\n"
        for node in self.get_nodes():
            representation += " "*4+str(node)+"\n"
            for link in node.get_linked_to():
                representation += " "*8+str(link)+"\n"

        return representation

graph = [
    [1, 2, 20],
    [2, 3, 2],
    [3, 4, 3],
    [4, 5, 2],
    [5, 6, 1],
    [6, 4, 3],
    [1, 4, 2],
    [3, 1, 3],
    [4, 3, 1],
    [1, 3, 2],
    [3, 5, 2],
    [5, 2, 2],
    [6, 2, 2],
    [7, 1, 1],
    [5, 7, 4],
    [6, 8, 3]
    ]

g = Graph(st = graph)

print('Інциденція:\n')
m = g.matrix_incident()
for n, i in enumerate(m):
    print(n+1, ":", i)


print('Суміжність:\n')
mm = g.matrix_adjacency()
for n, i in enumerate(mm):
    print(n+1, ":", i)

print(g)


print(g.get_nodes_power())
print(g.get_isolated())
#path = g.my_find_path(1, 2)
#print(path)
shortest_path = g.start_dejkstra(1, 2)
all_paths = g.find_all_paths(1, 2)
shortest_length = g.get_path_length(shortest_path)
print(shortest_length)
print(all_paths)
g.show(path = shortest_path, save_file = '1.png')
