'''
Heap Sort is a comparison-based sorting algorithm
that uses a binary heap data structure.
The algorithm works by first building a max heap
from the input array and then repeatedly extracting
the maximum element from the heap and placing it at the end of the array.
'''
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    #exists and left is greater
    if left < n and arr[largest] < arr[left]:
        largest = left
    #exists and right is greater
    if right < n and arr[largest] < arr[right]:
        largest = right
    #If tree is not follow heap property
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)
    #range(start, stop, step) ..starts from the last non-leaf node 
    #Build heap (rearrange array) max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    # One by one extract an element from heap and put it at last
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i] #Move current root to end
        heapify(arr, i, 0)              #call max heapify on the reduced heap


# Example Usage
arr = [12, 11, 13, 5, 6, 7]
heap_sort(arr)
print("Sorted array is:", arr)
