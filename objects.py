class Path:
    def __init__(self, path_list):
        pass

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

    def get_weight(self) -> int:
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

    def unmark_links(self) -> list:
        self.marked_links = []

    def mark_link(self, link) -> list:
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
        return s[offset]

    def add_link(self, to_id, weight, father) -> Link:
        new_link = Link(self.id, to_id, weight, father)
        self.links.append(new_link)
        return new_link

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Node {} Links count: {}".format(self.id, len(self.links))
