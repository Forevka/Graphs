# Graphs
My graphs implementation on python

## Features:
- Adjacency, Incidence, Distance Matrix

- Power of node and isolated node list

- Check Hamilton cycle in graph

- Bellman-Ford, Dijkstra, Floyd-Warshall path finding

- Topological sort

- BFS, DFS searches

- Showing graph object or path object with matplotlib

Code example:
```
from Graph.Graph import Graph
from Graph.exception import CantReach

graph = [
    [1, 2, -2],
    [1, 3, 4],
    [3, 2, 3],
    [3, 4, 1],
    [4, 3, 2],
    [5,5,None]
    ]

g = Graph(st = graph)
#print(g.fw_path(start = 4, end=3))
print('Інциденція:\n')
m = g.matrix_incidence()
for n, i in enumerate(m):
    print(n+1, ":", i)


print('Суміжність:\n')
mm = g.matrix_adjacency()
for n, i in enumerate(mm):
    print(n+1, ":", i)


hamilton = g.check_hamilton_cycle(1)
print("Без циклів" if hamilton is None else ("Цикл з вершини {} на шляху {}".format(hamilton[0], hamilton[1])))

print("Степені точок: ", g.get_nodes_power())
print("Ізольовані точки: ", g.get_isolated())
dejkstra_path = g.dejkstra_path(1, end = 2)

bellman_ford_path = g.bf_path(1)
topological_sorted_list = g.topological_sort()

print(dejkstra_path)
print(bellman_ford_path)
print("Топологічне сортування: ", topological_sorted_list)

bfs_p = list(g.bfs_paths(1, 2))
bfs_n = g.bfs(1)

dfs_p = list(g.dfs_paths(1, 2))
dfs_n = g.dfs(1)

#print("Довжина найкоротшого шляху:", dejkstra_path.get_length())

print("Всі можливі шляхи знайдені через BFS:\n", '\n'.join([str(i) for i in bfs_p]))
print("Всі вершини знайдені через BFS:", bfs_n)

print("Всі можливі шляхи знайдені через DFS:\n", '\n'.join([str(i) for i in dfs_p]))
print("Всі вершини знайдені через DFS:", dfs_n)
#g.show()
dejkstra_path.show()
```
![alt text](https://picua.org/images/2019/03/09/e05e7603b25e7a7c39423b7389a91b25.png)

## To do:
- [x] Make path object with .show() method
- [x] Module version
