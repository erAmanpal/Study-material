def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] > pivot:  # Note: For K'th largest, use '>' to partition
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickselect(arr, low, high, k):
    if low <= high:
        pi = partition(arr, low, high)
        if pi == k:
            return arr[pi]
        elif pi < k:
            return quickselect(arr, pi + 1, high, k)
        else:
            return quickselect(arr, low, pi - 1, k)
    return None

def find_kth_largest(arr, k):
    # Convert K'th largest to index
    indexk = k - 1
    return quickselect(arr, 0, len(arr) - 1, indexk)

# Example usage
arr = [12, 3, 5, 7, 19, 8, 15]
k = 3
print(f"The {k}'th largest element is {find_kth_largest(arr, k)}")
