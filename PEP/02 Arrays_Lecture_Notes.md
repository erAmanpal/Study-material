---

marp_content = '''
marp: true
theme: default
paginate: true
header: "Arrays - Placement Prep"
footer: "Data Structures & Algorithms"
style: |
  section {
    font-size: 20px;
  }
  h1 {
    color: #1a5276;
    font-size: 38px;
  }
  h2 {
    color: #2874a6;
    font-size: 30px;
  }
  h3 {
    color: #2e86c1;
    font-size: 24px;
  }
  code {
    font-size: 16px;
  }
  table {
    font-size: 16px;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .highlight {
    background: #fef9e7;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #f39c12;
  }
  .important {
    background: #fdedec;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #e74c3c;
  }
  .success {
    background: #eafaf1;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #27ae60;
  }
  .problem {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border: 2px solid #dee2e6;
    margin: 10px 0;
  }
  .hint {
    background: #e8f6f3;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #1abc9c;
  }
  .link-box {
    background: #eaf2f8;
    padding: 8px 12px;
    border-radius: 6px;
    display: inline-block;
    margin: 3px;
    font-size: 14px;
  }
---

<!-- _class: lead -->
# Arrays
## Complete Lecture Notes for Placement Preparation

**Data Structures & Algorithms**

---

# Table of Contents

1. Introduction to Arrays
2. Array Representation in Memory
3. Types of Arrays
4. Basic Array Operations
5. Time & Space Complexity of Array Operations
6. Array Traversal Techniques
7. Two Pointer Technique

---

8. Sliding Window Technique
9. Prefix Sum Technique
10. Kadane's Algorithm (Maximum Subarray)
11. Dutch National Flag Algorithm
12. Array Rotation Techniques
13. Practice Problems with Hints
14. Online Practice Problem Links
15. Interview Tips & Common Mistakes

---

# 1. Introduction to Arrays

## What is an Array?

An **array** is a collection of elements of the **same data type** stored in **contiguous memory locations**.

<div class="highlight">
<strong>Key Characteristics:</strong><br>
• Fixed size (in most languages) — size defined at creation<br>
• Contiguous memory allocation<br>
• Zero-based or one-based indexing<br>
• Random access — O(1) time to access any element<br>
• Homogeneous elements (same type)
</div>

---

## Why Arrays?

| Feature | Array | Linked List |
|---------|-------|-------------|
| Access | O(1) | O(n) |
| Insertion (beginning) | O(n) | O(1) |
| Insertion (end) | O(1)* |  O(n) |
| Memory | Contiguous | Scattered |
| Cache performance | Excellent | Poor |

*Amortized for dynamic arrays

---

# 1. Introduction to Arrays (Cont.)

## Array Declaration (Pseudocode/Python)

```python
# Static Array (fixed size)
arr = [0] * 10          # Size 10, initialized to 0

# Dynamic Array (resizable)
arr = []                # Empty list/array
arr.append(5)           # Grows automatically

# 2D Array (Matrix)
matrix = [[0] * m for _ in range(n)]  # n × m matrix
```

---

## Array Indexing (Python)

```python
arr = [10, 20, 30, 40, 50]

# Access
print(arr[0])    # 10  (First element)
print(arr[-1])   # 50  (Last element in Python)
print(arr[2])    # 30  (Third element)

# Index bounds: 0 to n-1 (for n elements)
# arr[n] → OUT OF BOUNDS ERROR!
```

---

# 2. Array Representation in Memory

## Memory Layout

```
Index:    0      1      2      3      4
        +------+------+------+------+------+
Value:  |  10  |  20  |  30  |  40  |  50  |
        +------+------+------+------+------+
Address: 1000   1004   1008   1012   1016
         (base)                           (base + 4×4)
```

---

## Address Calculation Formula

<div class="formula">
Address of arr[i] = Base Address + (i × Size of each element)
</div>

Example: If `base = 1000`, element size = `4 bytes`
- `arr[0]` → `1000 + 0×4 = 1000`
- `arr[3]` → `1000 + 3×4 = 1012`
- `arr[i]` → `1000 + i×4`

<div class="highlight">
<strong>Why O(1) Access?</strong> Direct address calculation - no traversal needed!
</div>

---

# 3. Types of Arrays

## By Dimension

| Type | Description | Example |
|------|-------------|---------|
| **1D Array** | Single row of elements | `[1, 2, 3, 4, 5]` |
| **2D Array** | Matrix / Grid | `[[1,2], [3,4], [5,6]]` |
| **3D Array** | Cube of elements | Used in 3D graphics, simulations |
| **Jagged Array** | Rows of varying lengths | `[[1,2], [3], [4,5,6]]` |


---

## By Size Behavior

| Type | Language | Resize? | Implementation |
|------|----------|---------|----------------|
| **Static Array** | C, Java | No | Fixed memory block |
| **Dynamic Array** | Python (list), Java (ArrayList), C++ (vector) | Yes | Doubling strategy |

**Exponential resizing strategy**: most commonly **doubling** —to manage its memory dynamically. 

---

## By Element Type

| Type | Example |
|------|---------|
| Integer Array | `[1, 2, 3, 4]` |
| Character Array (String) | `['a', 'b', 'c']` |
| Boolean Array | `[True, False, True]` |
| Object/Reference Array | Array of objects, pointers |

---

# 4. Basic Array Operations

## 1. Insertion

```python
# Insert at end (Dynamic Array)
arr = [1, 2, 3]
arr.append(4)        # [1, 2, 3, 4] — O(1) amortized

# Insert at beginning
arr.insert(0, 0)     # [0, 1, 2, 3, 4] — O(n)

# Insert at index i
arr.insert(2, 99)    # [0, 1, 99, 2, 3, 4] — O(n)
```

---

## 2. Deletion

```python
# Delete from end
arr.pop()            # O(1)

# Delete from beginning
arr.pop(0)           # O(n) — all elements shift left

# Delete at index i
arr.pop(2)           # O(n)

# Delete by value
arr.remove(99)       # O(n) — search + shift
```

---

## 3. Search

```python
# Linear Search
for i in range(len(arr)):
    if arr[i] == target:
        return i     # O(n)

# Binary Search (sorted array only)
# Use bisect module or implement manually — O(log n)
```

---

# 5. Time & Space Complexity of Array Operations

| Operation | Static Array | Dynamic Array | Notes |
|-----------|-------------|---------------|-------|
| Access by index | O(1) | O(1) | Direct address calculation |
| Search (unsorted) | O(n) | O(n) | Linear scan |
| Search (sorted) | O(log n) | O(log n) | Binary search |
| Insert at end | N/A | O(1) amortized | O(n) when resizing |
| Insert at beginning | O(n) | O(n) | All elements shift |

---

| Operation | Static Array | Dynamic Array | Notes |
|-----------|-------------|---------------|-------|
| Insert at index i | O(n) | O(n) | Elements shift right |
| Delete from end | O(1) | O(1) | No shifting |
| Delete from beginning | O(n) | O(n) | All elements shift |
| Delete at index i | O(n) | O(n) | Elements shift left |
| Resize | N/A | O(n) | Create new array, copy elements |
| Space | O(n) | O(n) | Contiguous memory |

---

# 6. Array Traversal Techniques

## Standard Traversal (Python)

```python
arr = [10, 20, 30, 40, 50]
for i in range(len(arr)):           # Forward traversal
    print(arr[i])                   # O(n)
for i in range(len(arr)-1, -1, -1): # Backward traversal
    print(arr[i])                   # O(n)
for element in arr:                 # Enhanced for loop (Python)
    print(element)                  # O(n)
for i, val in enumerate(arr):       # With index and value (Python)
    print(f"Index {i}: {val}")      # O(n)
```

---

## Traversal Patterns for Interviews

```python
# Skip every k elements
for i in range(0, len(arr), k):
    print(arr[i])          # O(n/k) = O(n)

# Traverse from both ends
left, right = 0, len(arr) - 1
while left <= right:
    print(arr[left], arr[right])
    left += 1
    right -= 1             # O(n)
```

---

# 7. Two Pointer Technique

## Concept

Use two pointers (indices) to traverse the array efficiently, usually from both ends or at different speeds.

<div class="highlight">
<strong>When to Use:</strong><br>
• Sorted array problems<br>
• Pair sum problems<br>
• Palindrome checking<br>
• Removing duplicates from sorted array<br>
• Merging two sorted arrays
</div>

---

## Example 1: Pair Sum in Sorted Array

```python
def pair_sum(arr, target):
    left = 0, right = len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return (left, right)
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None              # O(n) time, O(1) space
```
Similar Problem: https://www.geeksforgeeks.org/problems/key-pair5616/1

[Sorting + Two Pointers --OR-- HashSet ]
 

---

| Approach               | Time       | Space | Preferred       |
| ---------------------- | ---------- | ----- | --------------- |
| HashSet                | **O(N)**   | O(N)  | ✅ Most Expected |
| Sorting + Two Pointers | O(N log N) | O(1)  | Also Accepted   |

---
### Solution: with two pointer 
```java
import java.util.Arrays;
class Solution {
    boolean twoSum(int arr[], int target) {
        Arrays.sort(arr);
        int left = 0;
        int right = arr.length - 1;

        while (left < right) {
            int sum = arr[left] + arr[right];
            if (sum == target)  return true;

            if (sum < target)    left++;
            else    right--;
        }
        return false;
    }
}
```
---
### Solution: with hashset
```java
import java.util.HashSet;

class Solution {
    boolean twoSum(int arr[], int target) {
        HashSet<Integer> set = new HashSet<>();

        for (int num : arr) {
            int complement = target - num;

            if (set.contains(complement)) {
                return true;
            }

            set.add(num);
        }

        return false;
    }
}
```

---
## Example 2: Remove Duplicates from Sorted Array

```python
def remove_duplicates(arr):
    if not arr:
        return 0
    write = 1
    for read in range(1, len(arr)):
        if arr[read] != arr[read - 1]:
            arr[write] = arr[read]
            write += 1
    return write               # O(n) time, O(1) space
```

Similar Problem: https://www.geeksforgeeks.org/problems/remove-duplicate-elements-from-sorted-array/1 

---
### Solution:  Java Solution (Two Pointers)
```java
class Solution {
    ArrayList<Integer> removeDuplicates(int[] arr) {

        ArrayList<Integer> ans = new ArrayList<>();
        if (arr.length == 0)    return ans;
        ans.add(arr[0]);
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] != arr[i - 1]) {
                ans.add(arr[i]);
            }
        }
        return ans;
    }
}
```
**Time Complexity**: O(N)
**Space Complexity**: O(N) (for the returned ArrayList)

---
### Solution: Optimized In-Place Solution
```java
class Solution {
    int removeDuplicates(int[] arr) {

        if (arr.length == 0)    return 0;
        int j = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] != arr[j]) {
                j++;
                arr[j] = arr[i];
            }
        }
        return j + 1;
    }
}
```
**Time Complexity**: O(N)
**Space Complexity**: O(1)

---

# 8. Sliding Window Technique

## Concept

A subarray (window) that slides over the array to solve problems involving contiguous subarrays.

<div class="highlight">
<strong>When to Use:</strong><br>
• Maximum/minimum sum of subarray of size k<br>
• Longest substring with k distinct characters<br>
• Count of subarrays with sum = k<br>
• Fixed-size window problems
</div>


---

### Key Idea (Sliding Window)
- Compute the sum of the first k elements.
- For every new window:
    - Remove the element leaving the window.
    - Add the new element entering the window.
- Keep track of the maximum sum seen.


**Time Complexity**: O(N)
**Space Complexity**: O(1)

---

## Fixed Size Window

```python
def max_sum_subarray(arr, k):
    # Find maximum sum of any contiguous subarray of size k
    window_sum = sum(arr[:k])    # Initial window
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # Slide window
        max_sum = max(max_sum, window_sum)

    return max_sum                 # O(n) time, O(1) space
```
### Similar problem: https://www.geeksforgeeks.org/problems/max-sum-subarray-of-size-k5313/1 

---

### Solution Fixed Size Window: max-sum-subarray (Java version)
```java
class Solution {
    public int maximumSumSubarray(int[] arr, int k) {
        int n = arr.length;
        // Calculate sum of first window
        int windowSum = 0;
        for (int i = 0; i < k; i++) {
            windowSum += arr[i];
        }
        int maxSum = windowSum;
        // Slide the window
        for (int i = k; i < n; i++) {
            windowSum += arr[i] - arr[i - k];
            maxSum = Math.max(maxSum, windowSum);
        }
        return maxSum;
    }
}
```

---


## Variable Size Window

```python
def longest_subarray_with_sum_k(arr, k):
    # Find longest subarray with sum = k (positive numbers only)
    left = 0
    current_sum = 0
    max_length = 0

    for right in range(len(arr)):
        current_sum += arr[right]
        while current_sum > k:
            current_sum -= arr[left]
            left += 1
        if current_sum == k:
            max_length = max(max_length, right - left + 1)

    return max_length              # O(n) time, O(1) space
```
Similar Problem (using hashmap): https://www.geeksforgeeks.org/problems/longest-sub-array-with-sum-k0809/1 

---

# Pseudocode: longest-sub-array-with-sum
Expand the window by moving right. If the sum becomes too large, shrink it by moving left. Whenever the sum equals k, record the window length.
```
left = 0
sum = 0
maxLength = 0
for right = 0 to n-1
    sum = sum + arr[right]
    while sum > k
        sum = sum - arr[left]
        left++
    if sum == k
        maxLength = max(maxLength, right - left + 1)
return maxLength
```
---

### Solution Variable Size Window: longest-sub-array-with-sum (Java version)

```java
class Solution {
    public int longestSubarray(int[] arr, int k) {
        int left = 0, sum = 0, maxLen = 0;
        for (int right = 0; right < arr.length; right++) {
            sum += arr[right];  //Expand the window
            while (sum > k && left <= right) {  // Shrink the window while sum is greater than k
                sum -= arr[left];
                left++;
            }
            if (sum == k) { // Check if current window sum equals k
                maxLen = Math.max(maxLen, right - left + 1);
            }
        }
        return maxLen;
    }
}
```

---

# Problem with two pointer approach

- The Sliding Window (Two Pointers) approach works only when all array elements are non-negative (or strictly positive). 

- If negative numbers are present, this approach fails because expanding or shrinking the window no longer changes the sum predictably.

## Special Case: Only Positive Numbers

- If the array contains only positive integers, a sliding window (two pointers) solution also works in O(N) with O(1) extra space.

- However, since the problem states the array contains integers (which may include negative numbers), the Prefix Sum + HashMap approach is the correct and expected solution.

---
### Solution: with Hashmap + Prefix sum (cover under next topic)
```java
HashMap<Integer, Integer> map = new HashMap<>();
        int prefixSum = 0,  int maxLen = 0;
        for (int i = 0; i < arr.length; i++) {
            prefixSum += arr[i];
            // Case 1: Subarray starts from index 0
            if (prefixSum == k) {    maxLen = i + 1;    }
            // Case 2: Check if there exists a previous prefix sum
            if (map.containsKey(prefixSum - k)) {
                maxLen = Math.max(maxLen, i - map.get(prefixSum - k));
            }
            // Store the first occurrence of the prefix sum
            if (!map.containsKey(prefixSum)) {
                map.put(prefixSum, i);
            }
        }
        return maxLen;      // Space, Time Complexity O(N)
```

---
### Complete java code
```java
import java.util.HashMap;
public class Solution {
     public int longestSubarray(int[] arr, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int prefixSum = 0,maxLen = 0;
        for (int i = 0; i < arr.length; i++) {
            prefixSum += arr[i];
            if (prefixSum == k) {   // Case 1: Subarray starts from index 0
                maxLen = i + 1;
            }
            if (map.containsKey(prefixSum - k)) {   // Case 2: Check if there exists a previous prefix sum
                maxLen = Math.max(maxLen, i - map.get(prefixSum - k));            }
            if (!map.containsKey(prefixSum)) {  // Store the first occurrence of the prefix sum
                map.put(prefixSum, i);             }
        }
        return maxLen;
    }
    public static void main(String[] args) {
        Solution solution = new Solution();
        int[] arr = {10, 5, 2, 7, 1, 9};         int k = 15; // example array and target sum
        int result = solution.longestSubarray(arr, k);
        System.out.println("Longest subarray length with sum " + k + ": " + result);
    }
}
```


---

# 9. Prefix Sum Technique

## Concept

Precompute **__cumulative__** sums to answer range sum queries in O(1) time.

<div class="highlight">
<strong>When to Use:</strong><br>
• Range sum queries (multiple queries)<br>
• Subarray sum problems<br>
• Equilibrium index problems<br>
• Count of subarrays with given sum
</div>

---

## Building Prefix Sum
Build a Prefix Sum array as the preprocessing step:

```python
def build_prefix_sum(arr):
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

# prefix[i] = sum of arr[0] to arr[i-1]
# Sum of arr[l] to arr[r] = prefix[r+1] - prefix[l]
```
arr:  2 6 12 20 30
prefix sum: 0 2 6 12 20 30

---

## Range Sum Query

```python
def range_sum(prefix, l, r):
    return prefix[r + 1] - prefix[l]   # O(1) per query!

# Example:
# arr = [1, 2, 3, 4, 5]
# prefix = [0, 1, 3, 6, 10, 15]
# Sum of arr[1:3] = prefix[4] - prefix[1] = 10 - 1 = 9
```

---

# 9. Prefix Sum Technique (Cont.)

## 2D Prefix Sum (Matrix)

```python
def build_2d_prefix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]

    for i in range(rows):
        for j in range(cols):
            prefix[i+1][j+1] = (matrix[i][j]
                                + prefix[i][j+1]
                                + prefix[i+1][j]
                                - prefix[i][j])
    return prefix

# Sum of submatrix from (r1,c1) to (r2,c2):
# sum = prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
```

---

<div class="important">
<strong>Time Complexity:</strong><br>
• Building prefix: O(n) for 1D, O(rows × cols) for 2D<br>
• Each query: O(1)<br>
• Best when you have many queries on the same array
</div>

---

# 10. Kadane's Algorithm (Maximum Subarray)

## Problem
Find the maximum sum of any contiguous subarray.

## Algorithm

```python
def kadane(arr):
    max_so_far = arr[0]
    current_max = arr[0]

    for i in range(1, len(arr)):
        # Either extend the previous subarray or start new
        current_max = max(arr[i], current_max + arr[i])
        max_so_far = max(max_so_far, current_max)

    return max_so_far              # O(n) time, O(1) space
```

---

## How It Works

```
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

Step by step:
  i=0: current_max = -2, max_so_far = -2
  i=1: current_max = max(1, -2+1) = 1, max_so_far = 1
  i=2: current_max = max(-3, 1-3) = -2, max_so_far = 1
  i=3: current_max = max(4, -2+4) = 4, max_so_far = 4
  i=4: current_max = max(-1, 4-1) = 3, max_so_far = 4
  i=5: current_max = max(2, 3+2) = 5, max_so_far = 5
  i=6: current_max = max(1, 5+1) = 6, max_so_far = 6
  i=7: current_max = max(-5, 6-5) = 1, max_so_far = 6
  i=8: current_max = max(4, 1+4) = 5, max_so_far = 6

Result: 6 (subarray [4, -1, 2, 1])
```

---

### 10. Kadane's Algorithm (Cont.)

Variations: 1. Find the Subarray Itself

```python
def kadane_with_subarray(arr):
    max_so_far = arr[0]
    current_max = arr[0]
    start = end = temp_start = 0

    for i in range(1, len(arr)):
        if current_max + arr[i] < arr[i]:
            current_max = arr[i]
            temp_start = i
        else:
            current_max += arr[i]

        if current_max > max_so_far:
            max_so_far = current_max
            start = temp_start
            end = i
    return max_so_far, arr[start:end+1]   # O(n) time
```

---

### 2. Maximum Circular Subarray

```python
def max_circular_subarray(arr):
    # Case 1: Maximum subarray is non-circular (Kadane's)
    # Case 2: Maximum subarray is circular (wraps around)
    # Case 2 sum = Total sum - Minimum subarray sum

    total = sum(arr)
    max_kadane = kadane(arr)

    # Invert array and find min subarray
    inverted = [-x for x in arr]
    min_kadane = -kadane(inverted)

    # If all elements are negative
    if total == min_kadane:
        return max_kadane

    return max(max_kadane, total - min_kadane)   # O(n) time
```

---

# 11. Dutch National Flag Algorithm

## Problem
Sort an array containing only 0s, 1s, and 2s in O(n) time with O(1) space.

---
## Algorithm (Three-Way Partitioning)

```python
def dutch_national_flag(arr):
    low = mid = 0
    high = len(arr) - 1

    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:  # arr[mid] == 2
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    return arr                     # O(n) time, O(1) space
```

---

## How It Works

```
Regions:
[0 ... low-1]     → All 0s
[low ... mid-1]   → All 1s
[mid ... high]    → Unknown (to be processed)
[high+1 ... n-1]  → All 2s

Pointers:
  low  → Position to place next 0
  mid  → Current element being examined
  high → Position to place next 2
```

<div class="highlight">
<strong>Generalization:</strong> Can be extended to sort arrays with 3 distinct values or for 3-way quicksort partitioning.
</div>

---

# 12. Array Rotation Techniques

## 1. Using Extra Array (O(n) space)

```python
def rotate_left_extra(arr, k):
    n = len(arr)
    k = k % n                      # Handle k > n
    return arr[k:] + arr[:k]        # O(n) time, O(n) space
```

---

## 2. Juggling Algorithm (GCD based)

```python
import math

def rotate_left_juggling(arr, k):
    n = len(arr)
    k = k % n
    gcd = math.gcd(n, k)

    for i in range(gcd):
        temp = arr[i]
        j = i
        while True:
            next_idx = (j + k) % n
            if next_idx == i:
                break
            arr[j] = arr[next_idx]
            j = next_idx
        arr[j] = temp
    return arr                     # O(n) time, O(1) space
```
---

## 3. Reversal Algorithm (Most Elegant)

```python
def reverse(arr, start, end):
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1

def rotate_left_reversal(arr, k):
    n = len(arr)
    k = k % n
    reverse(arr, 0, k - 1)         # Reverse first k elements
    reverse(arr, k, n - 1)        # Reverse remaining n-k elements
    reverse(arr, 0, n - 1)        # Reverse entire array
    return arr                     # O(n) time, O(1) space
```

---

# 12. Array Rotation Techniques (Cont.)

## Rotation Summary Table

| Method | Time | Space | Code Complexity |
|--------|------|-------|-----------------|
| Extra Array | O(n) | O(n) | Simple |
| Juggling (GCD) | O(n) | O(1) | Complex |
| Reversal | O(n) | O(1) | Moderate |
| Cyclic Replacements | O(n) | O(1) | Moderate |

<div class="highlight">
<strong>Interview Tip:</strong> The Reversal Algorithm is the most elegant and commonly expected in interviews. Remember: "Reverse parts, then reverse whole."
</div>

---

### Finding Element in Rotated Sorted Array

```python
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1                      # O(log n) time, O(1) space
```

---

# 13. Practice Problems with Hints

## Problem 1: Reverse an Array

<div class="problem">
<strong>Problem:</strong> Reverse the given array in-place.<br>
<strong>Input:</strong> [1, 2, 3, 4, 5]<br>
<strong>Output:</strong> [5, 4, 3, 2, 1]
</div>

---

<div class="hint">
<strong>Hint:</strong> Use Two Pointer technique. Swap elements from both ends moving towards center.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def reverse_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 2: Find the Maximum Element

<div class="problem">
<strong>Problem:</strong> Find the maximum element in an unsorted array.<br>
<strong>Input:</strong> [3, 1, 4, 1, 5, 9, 2, 6]<br>
<strong>Output:</strong> 9
</div>

<div class="hint">
<strong>Hint:</strong> Single pass traversal. Keep track of running maximum.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def find_max(arr):
    max_val = arr[0]
    for num in arr[1:]:
        if num > max_val:
            max_val = num
    return max_val
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 3: Second Largest Element

<div class="problem">
<strong>Problem:</strong> Find the second largest element in an array.<br>
<strong>Input:</strong> [10, 20, 4, 45, 99]<br>
<strong>Output:</strong> 45
</div>

<div class="hint">
<strong>Hint:</strong> Track both largest and second largest in a single pass. Handle duplicates!<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

---

```python
def second_largest(arr):
    if len(arr) < 2:
        return None

    first = second = float('-inf')
    for num in arr:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num

    return second if second != float('-inf') else None
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 4: Check if Array is Sorted

<div class="problem">
<strong>Problem:</strong> Check if the array is sorted in non-decreasing order.<br>
<strong>Input:</strong> [1, 2, 2, 3, 4]<br>
<strong>Output:</strong> True
</div>

<div class="hint">
<strong>Hint:</strong> Compare each element with the next one. If any arr[i] > arr[i+1], return False.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def is_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 5: Remove Duplicates from Sorted Array

<div class="problem">
<strong>Problem:</strong> Remove duplicates from a sorted array in-place. Return the new length.<br>
<strong>Input:</strong> [1, 1, 2, 2, 3, 4, 4, 4, 5]<br>
<strong>Output:</strong> 5, array becomes [1, 2, 3, 4, 5, _, _, _, _]
</div>

---

<div class="hint">
<strong>Hint:</strong> Two Pointer technique. One pointer (write) for unique elements, another (read) to scan.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def remove_duplicates(arr):
    if not arr:
        return 0
    write = 1
    for read in range(1, len(arr)):
        if arr[read] != arr[read - 1]:
            arr[write] = arr[read]
            write += 1
    return write
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 6: Move Zeros to End

<div class="problem">
<strong>Problem:</strong> Move all zeros to the end while maintaining relative order of non-zero elements.<br>
<strong>Input:</strong> [0, 1, 0, 3, 12]<br>
<strong>Output:</strong> [1, 3, 12, 0, 0]
</div>

---

<div class="hint">
<strong>Hint:</strong> Two Pointer technique. Write pointer for non-zero elements, read pointer to scan. Fill remaining with zeros.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def move_zeros(arr):
    write = 0
    for read in range(len(arr)):
        if arr[read] != 0:
            arr[write] = arr[read]
            write += 1
    for i in range(write, len(arr)):
        arr[i] = 0
    return arr
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 7: Leaders in an Array

<div class="problem">
<strong>Problem:</strong> An element is a leader if it's greater than all elements to its right.<br>
<strong>Input:</strong> [16, 17, 4, 3, 5, 2]<br>
<strong>Output:</strong> [17, 5, 2]
</div>

---

<div class="hint">
<strong>Hint:</strong> Traverse from right to left. Keep track of maximum seen so far. If current > max, it's a leader.<br>
<strong>Expected:</strong> O(n) time, O(1) space (excluding output)
</div>

```python
def find_leaders(arr):
    n = len(arr)
    leaders = []
    max_from_right = float('-inf')

    for i in range(n - 1, -1, -1):
        if arr[i] > max_from_right:
            leaders.append(arr[i])
            max_from_right = arr[i]

    return leaders[::-1]           # Reverse to maintain original order
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 8: Maximum Subarray Sum (Kadane's)

<div class="problem">
<strong>Problem:</strong> Find the maximum sum of any contiguous subarray.<br>
<strong>Input:</strong> [-2, 1, -3, 4, -1, 2, 1, -5, 4]<br>
<strong>Output:</strong> 6 (subarray [4, -1, 2, 1])
</div>

---

<div class="hint">
<strong>Hint:</strong> At each position, decide whether to extend the previous subarray or start a new one. Use Kadane's algorithm.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def max_subarray_sum(arr):
    max_so_far = current_max = arr[0]
    for i in range(1, len(arr)):
        current_max = max(arr[i], current_max + arr[i])
        max_so_far = max(max_so_far, current_max)
    return max_so_far
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 9: Equilibrium Index

<div class="problem">
<strong>Problem:</strong> Find an index where sum of elements before it equals sum of elements after it.<br>
<strong>Input:</strong> [-7, 1, 5, 2, -4, 3, 0]<br>
<strong>Output:</strong> 3 (index 3: -7+1+5 = -4+3+0 = -1)
</div>

---

<div class="hint">
<strong>Hint:</strong> Use prefix sum. Total sum - prefix_sum[i] - arr[i] == prefix_sum[i]. Or use two pointers from both ends.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def equilibrium_index(arr):
    total = sum(arr)
    left_sum = 0
    for i in range(len(arr)):
        if left_sum == total - left_sum - arr[i]:
            return i
        left_sum += arr[i]
    return -1
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 10: Subarray with Given Sum

<div class="problem">
<strong>Problem:</strong> Find a contiguous subarray with sum equal to target (positive numbers only).<br>
<strong>Input:</strong> [1, 4, 20, 3, 10, 5], target = 33<br>
<strong>Output:</strong> [20, 3, 10] (indices 2 to 4)
</div>

---

<div class="hint">
<strong>Hint:</strong> Sliding Window technique. Expand window by adding from right, shrink from left if sum exceeds target.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def subarray_with_sum(arr, target):
    left = current_sum = 0
    for right in range(len(arr)):
        current_sum += arr[right]
        while current_sum > target and left <= right:
            current_sum -= arr[left]
            left += 1
        if current_sum == target:
            return arr[left:right+1]
    return None
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 11: Merge Two Sorted Arrays

<div class="problem">
<strong>Problem:</strong> Merge two sorted arrays into one sorted array.<br>
<strong>Input:</strong> [1, 3, 5, 7], [2, 4, 6, 8]<br>
<strong>Output:</strong> [1, 2, 3, 4, 5, 6, 7, 8]
</div>

---

<div class="hint">
<strong>Hint:</strong> Two Pointer technique. Compare elements at both pointers, add smaller to result, advance that pointer.<br>
<strong>Expected:</strong> O(n+m) time, O(n+m) space (for result)
</div>

```python
def merge_sorted(arr1, arr2):
    result = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 12: Majority Element (Boyer-Moore Voting)

<div class="problem">
<strong>Problem:</strong> Find element that appears more than n/2 times.<br>
<strong>Input:</strong> [3, 3, 4, 2, 4, 4, 2, 4, 4]<br>
<strong>Output:</strong> 4
</div>

---

<div class="hint">
<strong>Hint:</strong> Boyer-Moore Voting Algorithm. Maintain a candidate and count. Cancel out pairs of different elements.<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def majority_element(arr):
    candidate = None
    count = 0
    for num in arr:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1
    # Verify candidate (optional if guaranteed to exist)
    return candidate if arr.count(candidate) > len(arr) // 2 else None
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 13: Trapping Rain Water

<div class="problem">
<strong>Problem:</strong> Given elevation map, compute trapped water after raining.<br>
<strong>Input:</strong> [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]<br>
<strong>Output:</strong> 6
</div>

---

<div class="hint">
<strong>Hint:</strong> Two Pointer approach. Water at each position = min(max_left, max_right) - height[i]. Track max from left and right.
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def trap_rain_water(height):
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 14: Product of Array Except Self

<div class="problem">
<strong>Problem:</strong> Return array where each element is product of all other elements. No division allowed.<br>
<strong>Input:</strong> [1, 2, 3, 4]<br>
<strong>Output:</strong> [24, 12, 8, 6]
</div>

---

<div class="hint">
<strong>Hint:</strong> Use prefix and suffix products. For each i, result[i] = product of all elements before i × product of all after i.
<strong>Expected:</strong> O(n) time, O(1) extra space (excluding output)
</div>

```python
def product_except_self(arr):
    n = len(arr)
    result = [1] * n
    # Prefix products
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= arr[i]
    # Suffix products
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= arr[i]

    return result
```

---

# 13. Practice Problems with Hints (Cont.)

## Problem 15: Find Missing Number

<div class="problem">
<strong>Problem:</strong> Array contains n distinct numbers from 0 to n. Find the missing one.<br>
<strong>Input:</strong> [3, 0, 1]<br>
<strong>Output:</strong> 2
</div>

---

<div class="hint">
<strong>Hint:</strong> XOR all array elements with all numbers from 0 to n. XOR of same numbers cancels out. Or use sum formula: n(n+1)/2 - sum(arr).<br>
<strong>Expected:</strong> O(n) time, O(1) space
</div>

```python
def find_missing_xor(arr):
    n = len(arr)
    missing = n
    for i in range(n):
        missing ^= (i + 1) ^ arr[i]
    return missing

def find_missing_sum(arr):
    n = len(arr)
    expected = n * (n + 1) // 2
    return expected - sum(arr)
```

---

# 14. Online Practice Problem Links

## HackerRank Array Problems

<div class="link-box">Arrays - DS</div>
<div class="link-box">2D Array - DS</div>
<div class="link-box">Dynamic Array</div>
<div class="link-box">Left Rotation</div>
<div class="link-box">Sparse Arrays</div>
<div class="link-box">Array Manipulation</div>

**URL:** https://www.hackerrank.com/domains/data-structures/arrays

---

## LeetCode Array Problems (Top 150)

<div class="link-box">#1 Two Sum</div>
<div class="link-box">#121 Best Time to Buy/Sell Stock</div>
<div class="link-box">#53 Maximum Subarray (Kadane's)</div>
<div class="link-box">#238 Product Except Self</div>
<div class="link-box">#152 Maximum Product Subarray</div>
<div class="link-box">#153 Find Min in Rotated Sorted Array</div>
<div class="link-box">#33 Search in Rotated Sorted Array</div>
<div class="link-box">#15 3Sum</div>
<div class="link-box">#11 Container With Most Water</div>
<div class="link-box">#42 Trapping Rain Water</div>

**URL:** https://leetcode.com/tag/array/

---

# 14. Online Practice Problem Links (Cont.)

### GeeksforGeeks Array Problems

<div class="link-box">Reverse Array</div>
<div class="link-box">Find Min & Max</div>
<div class="link-box">Kth Smallest/Largest</div>
<div class="link-box">Sort 0s, 1s, 2s (Dutch Flag)</div>
<div class="link-box">Move Negatives to One Side</div>
<div class="link-box">Union & Intersection</div>
<div class="link-box">Cyclically Rotate Array</div>
<div class="link-box">Largest Sum Contiguous Subarray</div>
<div class="link-box">Minimize Heights</div>
<div class="link-box">Merge Without Extra Space</div>

**URL:** https://www.geeksforgeeks.org/array-data-structure/

---

## CodeChef Array Problems (Beginner → Intermediate)

<div class="link-box">Search an Element</div>
<div class="link-box">Max Sum Subarray</div>
<div class="link-box">Arrays, Strings & Sorting</div>
<div class="link-box">INTEST (Enormous Input Test)</div>
<div class="link-box">FLOW006 (Sum of Digits)</div>

**URL:** https://www.codechef.com/practice/course/arrays/ARRAYS

---

# 14. Online Practice Problem Links (Cont.)

## Codeforces Array Problems

<div class="link-box">A. Array with Odd Sum</div>
<div class="link-box">A. Array Coloring</div>
<div class="link-box">B. Array Reordering</div>
<div class="link-box">A. Array Balancing</div>
<div class="link-box">C. Array Game</div>

**URL:** https://codeforces.com/problemset?tags=arrays

---

## InterviewBit Array Problems

<div class="link-box">Pascal Triangle</div>
<div class="link-box">Next Permutation</div>
<div class="link-box">Kth Row of Pascal's Triangle</div>
<div class="link-box">Anti Diagonals</div>
<div class="link-box">Noble Integer</div>
<div class="link-box">Pick from Both Sides</div>

**URL:** https://www.interviewbit.com/courses/programming/topics/arrays/

---

# 15. Interview Tips & Common Mistakes

## Before the Interview

<div class="highlight">
✅ <strong>Master these patterns:</strong><br>
• Two Pointer (sorted arrays, pair sum, palindrome)<br>
• Sliding Window (subarray problems, fixed/variable size)<br>
• Prefix Sum (range queries, subarray sum)<br>
• Kadane's Algorithm (maximum subarray)<br>
• Dutch National Flag (3-way partitioning)<br>
• Binary Search on Arrays (sorted, rotated)<br>
• Boyer-Moore Voting (majority element)
</div>

---

## During the Interview

<div class="important">
1. <strong>Always clarify:</strong> Is the array sorted? Can it contain duplicates? Negative numbers?

2. <strong>State complexity upfront:</strong> "I'll solve this in O(n) time and O(1) space"
3. <strong>Handle edge cases:</strong> Empty array, single element, all same elements
4. <strong>Walk through with an example:</strong> Don't just write code, explain your logic
5. <strong>Optimize if possible:</strong> "Can we do better than O(n²)? Yes, using hash map..."
</div>

---

# 15. Interview Tips & Common Mistakes (Cont.)

## Common Mistakes to Avoid

<div class="important">
❌ <strong>Off-by-one errors:</strong> Confusing 0-based and 1-based indexing<br>
❌ <strong>Array bounds:</strong> Accessing arr[n] when valid indices are 0 to n-1<br>
❌ <strong>Integer overflow:</strong> Sum of large numbers exceeding int range<br>
❌ <strong>Modifying while iterating:</strong> Changing array size during loop<br>
❌ <strong>Forgetting to handle empty array:</strong> Always check if len(arr) == 0<br>
❌ <strong>Not asking about duplicates:</strong> Solution changes if duplicates exist<br>
❌ <strong>Using O(n) space when O(1) is possible:</strong> Always think about optimization
</div>

---

## Quick Complexity Reference for Arrays

| Operation | Best Case | Average | Worst Case |
|-----------|-----------|---------|------------|
| Access | O(1) | O(1) | O(1) |
| Search (unsorted) | O(1) | O(n) | O(n) |
| Search (sorted) | O(1) | O(log n) | O(log n) |
| Insert at end | O(1) | O(1) | O(n) |
| Insert at middle | O(1) | O(n) | O(n) |
| Delete | O(1) | O(n) | O(n) |

---

<!-- _class: lead -->
# Thank You!
## Keep Practicing Arrays Daily! 🎯

**Key Takeaways:**
- Arrays are the foundation — master them first
- Learn the 7 core patterns (Two Pointer, Sliding Window, Prefix Sum, etc.)
- Practice on multiple platforms (LeetCode, GFG, HackerRank, CodeChef)
- Always analyze time and space complexity
- Handle edge cases in every solution

---

**Recommended Practice Order:**
1. Basic traversal problems → 2. Two Pointer → 3. Sliding Window → 4. Prefix Sum → 5. Kadane's → 6. Binary Search on Arrays → 7. Advanced (Trapping Rain, Product Except Self)

