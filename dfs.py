from collections import defaultdict

class GraphDFS:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # undirected

    def dfs(self, node, visited=None):
        if visited is None:
            visited = set()
        visited.add(node)
        print(node, end=' ')
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

# Example usage
g = GraphDFS()
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]

for u, v in edges:
    g.add_edge(u, v)

print("DFS Traversal (Recursive):")
g.dfs(0)
