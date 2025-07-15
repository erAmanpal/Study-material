class MaxHeap:
    def __init__(self):
        self.heap = []
# Find parent
    def parent(self, i):
        return (i - 1) // 2
#Find Left child
    def left_child(self, i):
        return 2 * i + 1
# Find Right child
    def right_child(self, i):
        return 2 * i + 2
# Insert new node
    def insert(self, key):
        self.heap.append(key)
        self.heapify_up(len(self.heap) - 1)
# Move new node to exact location
    def heapify_up(self, i):
        while i != 0 and self.heap[self.parent(i)] < self.heap[i]:
            self.heap[self.parent(i)], self.heap[i] = self.heap[i], self.heap[self.parent(i)]
            i = self.parent(i)
# Delete  the max node
    def extract_max(self):
        if len(self.heap) == 0:
            return None
        root = self.heap[0]
        #Replace root with last element
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return root
# Adjust the tree after deletion
    def heapify_down(self, i):
        largest = i
        left = self.left_child(i)
        right = self.right_child(i)
        #Find the chaild having larger value 
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        #Swap if child is greater
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self.heapify_down(largest)
# Show the heap list
    def display(self):
        print(self.heap)


# Example Usage
max_heap = MaxHeap()
max_heap.insert(3)
max_heap.insert(2)
#max_heap.insert(15)
#max_heap.insert(5)
#max_heap.insert(4)
#max_heap.insert(45)

print("Max Heap: ")
max_heap.display()

print("Extracted Max: ", max_heap.extract_max())

print("Max Heap after extraction: ")
max_heap.display()
