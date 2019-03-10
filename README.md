# Graphs
My graphs representation on python

## Features:
- Adjacency, Incidency Matrix

- Power of node and isolated node list

- Bellman-Ford, Dijkstra path findng

- BFS, DFS searches

Code example:
```
from graph_class import Graph
from exception import CantReach

graph = [
    [1, 2, 13],
    [1, 6, 2],
    [2, 3, 2],
    [3, 5, 3],
    [4, 8, 3],
    [4, 2, 2],
    [5, 2, 1],
    [6, 1, 3],
    [6, 7, 2],
    [7, 8, 1],
    [6, 2, 2],
    [6, 3, -3],
    [6, 4, 1],
    [6, 5, 2],
    [5, 1, 4]
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

#print(g)

print("Степені точок: ", g.get_nodes_power())
print("Ізольовані точки: ", g.get_isolated())
dejkstra_path = g.start_dejkstra(1, 4)

bellman_ford_path = g.reconstruct_bf_path(1, 2)
bellman_ford_paths = g.bellman_ford(1)

bfs_p = list(g.bfs_paths(1, 2))
bfs_n = g.bfs(1)

dfs_p = list(g.dfs_paths(1, 2))
dfs_n = g.dfs(1)

print("Найкоротший шлях знайдений через алгоритм Дейкстри: ", dejkstra_path)

print("Всі можливі шляхи знайдені через Бельмана-Форда", bellman_ford_paths)
print("Найкоротший шлях знайдений через Бельмана-Форда", bellman_ford_path)

print("Довжина найкоротшого шляху:", dejkstra_path.get_length())

print("Всі можливі шляхи знайдені через BFS:\n", '\n'.join([str(i) for i in bfs_p]))
print("Всі вершини знайдені через BFS:", bfs_n)

print("Всі можливі шляхи знайдені через DFS:\n", '\n'.join([str(i) for i in dfs_p]))
print("Всі вершини знайдені через DFS:", dfs_n)

bellman_ford_path()
```
![alt text](https://picua.org/images/2019/03/09/e05e7603b25e7a7c39423b7389a91b25.png)

## To do:
- [x] Make path object with .visualize() method
