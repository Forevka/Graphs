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
        #pr(self.father_graph)

    def get_linked_from(self) -> Link:
        all_links = father.get_links()
        for link in all_links:
            if link.from_id == self.id:
                yield link

    def get_linked_to(self) -> Link:# pass
        for i in self.links:
            yield i

    def get_cheapest_link(self, offset) -> Link:
        s = sorted(list(self.get_linked_to()), key=getKey)
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

    def get_isolated(self):
        nodes = self.get_nodes()
        l = dict((num, 0) for num in range(1, len(nodes)+1, 1))
        for node in nodes:
            for link in node.get_linked_to():
                if link.from_id != link.to_id:
                    l.update({link.from_id: l.get(link.from_id, 0)+1})

        return [i for i in l.keys() if l[i]==0]

    def find_path(self, from_id, to_id):
        '''not realized'''
        stack = []
        stack.append(from_id)
        this_node = self.get_all_nodes()[from_id]
        print(this_node)
        finded = False

        while(from_id != to_id):
            current_link = this_node.get_cheapest_link(0)
            print(current_link)
            new_node = current_link.get_node_to()
            print(new_node)
            return self.find_path(new_node.id, to_id)
            #finded = True



    def show(self, save_path = None):
        G=nx.MultiDiGraph()
        fig, ax = plt.subplots()
        nodes = self.get_nodes()
        links = self.get_links()
        labels = {}
        for n, node in enumerate(nodes):
            G.add_node(node.id)
            labels[n+1] = str(node.id)#get_literal(node.id)#str(node.id)
        for link in links:
            G.add_edge(link.from_id, link.to_id)

        pos=nx.circular_layout(G)
        nx.draw_networkx_labels(G, pos, labels ,font_size=11)

        ax=plt.gca()
        draw_network(G,pos,ax, links)
        ax.autoscale()
        if save_path is not None:
            print('saving')
            plt.savefig(save_path)

        plt.show()

    def __str__(self):
        representation = "Graph with:\n"
        for node in self.get_nodes():
            representation += " "*4+str(node)+"\n"
            for link in node.get_linked_to():
                representation += " "*8+str(link)+"\n"

        return representation

graph = [
    [1, 4, 5],
    [2, 1, 2],
    [3, 2, 3],
    [3, 4, 2],
    [2, 3, 4],
    [4, 1, 5],
    [1, 5, 2],
    [2, 5, 2],
    [3, 5, 2],
    [4, 5, 3],
    [5, 5, 2],
    [1, 6, 7]
    ]

g = Graph()

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
print(g.find_path(1, 5))
g.show('1.png')
