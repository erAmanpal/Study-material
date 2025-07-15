#import deque from collection package
from collections import deque
# BFS from given source s
def bfs(adj, s, visited):
    q = deque()         # Create a queue for BFS
    visited[s] = True   # Mark s node visited and enqueue it
    q.append(s)
    while q:            # Iterate over the queue
        curr = q.popleft()  # Dequeue a vertex from queue and print it
        print(curr, end=" ")
        # Get all adjacent vertices of the dequeued vertex. If an adjacent
        #has not been visited,mark it visited and enqueue it
        for x in adj[curr]:
            if not visited[x]:
                visited[x] = True
                q.append(x)
# Function to add an edge to the graph
def add_edge(adj, u, v):
    adj[u].append(v)
    adj[v].append(u)

if __name__ == "__main__":
    V = 5    # Number of vertices in the graph
    # Adjacency list representation of the graph
    adj = [[] for _ in range(V)]
    add_edge(adj, 0, 1)   # Add edges to the graph
    add_edge(adj, 0, 2)
    add_edge(adj, 1, 3)
    add_edge(adj, 1, 4)
    add_edge(adj, 2, 4)
    visited = [False] * V   # Mark all the vertices as not visited
    print("BFS starting from 0: ")
    bfs(adj, 0, visited)
'''
0----1----3
|    |   
|    4
|    |
2----
'''
