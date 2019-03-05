import numpy as np

class Node:
    def __init__(self, id):
        self.id = int(id)
        self.links_list = []

    def get_linked_to(self):# pass
        for i in self.links_list:
            yield i

    def add_link(self, to_id):
        new_link = Link(self.id, int(to_id))
        self.links_list.append(new_link)
        return new_link

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Graph {} ID Links count: {}".format(self.id, len(self.links_list))

class Link:
    def __init__(self, from_id, to_id):
        self.from_id = from_id
        self.to_id = to_id

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Link from {} to {}".format(self.from_id, self.to_id)

class Graph:
    def __init__(self, file = 'first.txt'):
        self.file = file
        self.nodes_list, self.links_list = self.load_graph()

    def create_node(self, temp_storage, node_id, link_to):
        node = temp_storage['nodes'].get(node_id)
        to_node = temp_storage['nodes'].get(link_to)
        if to_node is None:
            to_node = Node(link_to)
            temp_storage['nodes'].update({link_to: to_node})

        if node is None:
            node = Node(node_id)

        link = node.add_link(link_to)
        temp_storage['links'].append(link)
        temp_storage['nodes'].update({node_id: node})
        return node

    def load_graph(self):
        t = {'nodes':{}, 'links':[]}
        for line in open(self.file, 'r', encoding = 'utf8').readlines():
            line = line.replace('\n', '')
            id, to_id = line.split(' ')[0], line.split(' ')[1]
            node = self.create_node(t, id, to_id)

        return t['nodes'], t['links']

    def get_links(self):
        return self.links_list

    def get_nodes(self):
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

    def get_nodes_power(self):
        l = {}
        for node in self.nodes_list.values():
            for link in node.get_linked_to():
                print(link)
                l.update({link.from_id: l.get(link.from_id, 0)+1})
                l.update({link.to_id: l.get(link.to_id, 0)+1})

        return l

    def get_isolated(self):
        pass

g = Graph()
m = g.matrix_incident()

print('Інциденція:\n')
for n, i in enumerate(m):
    print(n+1, ":", i)


print('Суміжність:\n')
mm = g.matrix_adjacency()

for n, i in enumerate(mm):
    print(n+1, ":", i)

#print(g.get_nodes_power())
№print(g.get_isolated())
