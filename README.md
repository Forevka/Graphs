# Graphs
My graphs representation on python

## Features:
- Adjacency, Incidency Matrix

- Power of node and isolated node list

- Bellman-Ford, Dijkstra path findng

- BFS, DFS searches

Code example:
```
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

print('Incidency:\n')
m = g.matrix_incident()
for n, i in enumerate(m):
    print(n+1, ":", i)


print('Adjacency:\n')
mm = g.matrix_adjacency()
for n, i in enumerate(mm):
    print(n+1, ":", i)

#Reprsent current graph
print(g)


print(g.get_nodes_power())
print(g.get_isolated())

#Dejkstra and Bellman Ford
shortest_path_d = g.start_dejkstra(1, 2)
shortest_length_d = g.get_path_length(shortest_path)

all_paths_b = g.bellman_ford(1)
shortest_path_b = g.reconstruct_bf_path(1, 2)

#Breadth and Depth first searches
bfs_p = list(g.bfs_paths(1, 2))
bfs_n = g.bfs(1)

dfs_p = list(g.dfs_paths(1, 2))
dfs_n = g.dfs(1)

#Show finded path
g.show(path = shortest_path_d, save_file = '1.png')
```
![alt text](https://picua.org/images/2019/03/09/e05e7603b25e7a7c39423b7389a91b25.png)

## To do:
- [ ] Make path object with .visualize() method
