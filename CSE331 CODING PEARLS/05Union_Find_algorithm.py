class UnionFind:
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def __repr__(self):
        return f"UnionFind(\nparent={self.parent}, \n  rank={self.rank})"

# Example usage:
if __name__ == "__main__":
    elements = ['a', 'b', 'c', 'd', 'e']
    uf = UnionFind(elements)  # Create a Union-Find data structure with given elements
    
    # Perform some union operations
    uf.union('a', 'b')
    uf.union('b', 'd')
    
    # Check connectivity
    print(uf.connected('a', 'b'))  # True, because 'a' is connected to 'b', which is connected to 'd'
    print(uf.connected('a', 'e'))  # False, because 'a' and 'e' are in different subsets
    
    # Print the internal state
    print(uf)
