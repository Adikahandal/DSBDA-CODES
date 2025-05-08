from collections import defaultdict, deque

class GraphBFS:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # undirected

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            print(node, end=' ')
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

# Example usage
g = GraphBFS()
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

for u, v in edges:
    g.add_edge(u, v)

print("BFS Traversal (Iterative):")
g.bfs(0)
