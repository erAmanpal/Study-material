---
marp: true
theme: default

paginate: true
header: "Asymptotic Notation - Placement Prep"
footer: "Data Structures & Algorithms"
style: |
  section {
    font-size: 22px;
  }
  h1 {
    color: #1a5276;
    font-size: 40px;
  }
  h2 {
    color: #2874a6;
    font-size: 32px;
  }
  code {
    font-size: 18px;
  }
  table {
    font-size: 18px;
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
  .formula {
    background: #eaf2f8;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
  }
---

<!-- _class: lead -->
# Asymptotic Notation
## Lecture Notes for Placement Preparation

**Data Structures & Algorithms**

---

# Table of Contents

1. Introduction to Asymptotic Analysis
2. Big-O Notation (O)
3. Big-Omega Notation (Ω)
4. Big-Theta Notation (Θ)
5. Little-o and Little-omega Notations
6. Common Complexity Classes
7. Analyzing Time Complexity
8. Space Complexity Analysis
9. Recurrence Relations & Master Theorem
10. Amortized Analysis
11. Placement-Ready Problem Solving
12. Quick Reference Cheat Sheet

---

# 1. Introduction to Asymptotic Analysis

## Why Asymptotic Notation?

- **Problem**: Comparing algorithm efficiency by running them is impractical
  - Depends on hardware, programming language, compiler optimizations
  - Time-consuming for large inputs

- **Solution**: Mathematical analysis of algorithm behavior as input size grows
  - Focus on **growth rate**, not exact running time
  - Input size `n → ∞`

## Key Idea
> "How does the running time grow when input size becomes very large?"

---

# 1. Introduction to Asymptotic Analysis (Cont.)

## What We Ignore

| Factor | Why We Ignore It |
|--------|------------------|
| Constant factors | `2n` and `5n` both grow linearly |
| Lower-order terms | `n² + n` is dominated by `n²` for large `n` |
| Hardware speed | Same algorithm runs faster on better hardware |
| Programming language | Same logic, different constants |

## What We Care About
- **Rate of growth** as `n → ∞`
- **Worst-case**, **Average-case**, and **Best-case** scenarios

---

# 2. Big-O Notation (O) — Upper Bound

## Definition

<div class="formula">
f(n) = O(g(n)) if ∃ constants c > 0, n₀ ≥ 0 such that
0 ≤ f(n) ≤ c·g(n) for all n ≥ n₀
</div>

## Intuition
- `f(n)` grows **no faster than** `g(n)` and  `g(n)` is an **upper bound** on `f(n)`
---
## Visual Representation

![alt text](image.png)

---

# 2. Big-O Notation — Examples

## Common Examples

| Function f(n) | Big-O | Explanation |
|-------------|-------|-------------|
| `3n + 2` | O(n) | Drop constants and lower terms |
| `n² + 5n + 100` | O(n²) | n² dominates for large n |
| `2ⁿ + n³` | O(2ⁿ) | Exponential dominates polynomial |
| `log(n) + 5` | O(log n) | Constant is negligible |
| `100` | O(1) | Constant time |

---

## Properties
- **Transitivity**: If f(n) = O(g(n)) and g(n) = O(h(n)), then f(n) = O(h(n))
- **Reflexivity**: f(n) = O(f(n))
- **Sum Rule**: O(f(n)) + O(g(n)) = O(max(f(n), g(n)))
- **Product Rule**: O(f(n)) × O(g(n)) = O(f(n) × g(n))

---

# 2. Big-O Notation — Placement Tricks

## Dropping Constants & Lower Terms

<div class="highlight">
<strong>Rule of Thumb:</strong> Keep only the fastest-growing term, drop its coefficient.
</div>

```python
# Example: Simplify these to Big-O
f(n) = 5n³ + 3n² + 100n + 50     → O(n³)
f(n) = 2ⁿ + n¹⁰⁰                  → O(2ⁿ)
f(n) = n·log(n) + n               → O(n·log n)
f(n) = log(n!)                    → O(n·log n)  [Stirling's approximation]
```

## Common Mistakes in Interviews

<div class="important">
❌ O(2n) is wrong — write O(n)<br>
❌ O(n + log n) is wrong — write O(n)<br>
❌ O(1) + O(n) is wrong — write O(n)<br>
✅ Always simplify to the cleanest form
</div>

---

# 3. Big-Omega Notation (Ω) : Lower Bound

## Definition

<div class="formula">
f(n) = Ω(g(n)) if ∃ constants c > 0, n₀ ≥ 0 such that<br>
0 ≤ c·g(n) ≤ f(n) for all n ≥ n₀
</div>

## Intuition
- `f(n)` grows **at least as fast as** `g(n)`
- `g(n)` is a **lower bound** on `f(n)`

---

## Visual Representation

![alt text](image-1.png)


---
## Example
- `n² + 5n = Ω(n²)` ✓
- `n² + 5n = Ω(n)` ✓ (also valid, but loose)
- `n² + 5n = Ω(n³)` ✗

---


# 4. Big-Theta Notation (Θ) - Tight Bound

## Definition

<div class="formula">
f(n) = Θ(g(n)) if ∃ constants c₁, c₂ > 0, n₀ ≥ 0 such that<br>
0 ≤ c₁·g(n) ≤ f(n) ≤ c₂·g(n) for all n ≥ n₀
</div>

## Intuition
- `f(n)` grows **at the same rate** as `g(n)`
- Both upper and lower bounds are `g(n)`

<div class="highlight">
<strong>Key Relationship:</strong> f(n) = Θ(g(n)) ⟺ f(n) = O(g(n)) AND f(n) = Ω(g(n))
</div>

---

## Example
- `3n² + 5n + 10 = Θ(n²)` ✓
- `3n² + 5n + 10 = Θ(n³)` ✗
- `3n² + 5n + 10 = Θ(n)` ✗

---

# 4. Big-Theta - When to Use

## In Placement Interviews

<div class="highlight">
<strong>Best Practice:</strong> Use Θ when you have a tight bound.<br>
Use O when you only have an upper bound (common for worst-case analysis).
</div>

| Scenario | Notation to Use |
|----------|----------------|
| Worst-case analysis (upper bound) | O |
| Best-case analysis (lower bound) | Ω |
| Exact growth rate known | Θ |
| Average-case analysis | Usually Θ or O |


---

## Example Question
> "What is the time complexity of Merge Sort?"
- Answer: **Θ(n log n)** in all cases (best, average, worst)

> "What is the time complexity of Quick Sort?"
- Answer: **O(n log n)** average, **O(n²)** worst case

---

# 5. Little-o and Little-omega Notations

## Little-o: Strictly Smaller Growth

<div class="formula">
f(n) = o(g(n)) if ∀ c > 0, ∃ n₀ such that<br>
0 ≤ f(n) < c·g(n) for all n ≥ n₀
</div>

- `f(n)` grows **strictly slower** than `g(n)`
- **Not** just "no faster" — it's "strictly slower"

## Little-omega: Strictly Larger Growth

<div class="formula">
f(n) = ω(g(n)) if ∀ c > 0, ∃ n₀ such that<br>
0 ≤ c·g(n) < f(n) for all n ≥ n₀
</div>

- `f(n)` grows **strictly faster** than `g(n)`

---

# 5. Little-o and Little-omega — Examples

## Examples

| Relationship | Result | Explanation |
|-------------|--------|-------------|
| `n = o(n²)` | ✓ | Linear < Quadratic |
| `n = o(n)` | ✗ | Not strictly smaller |
| `n² = ω(n)` | ✓ | Quadratic > Linear |
| `n² = ω(n²)` | ✗ | Not strictly larger |
| `log n = o(n)` | ✓ | Logarithmic < Linear |
| `2ⁿ = ω(n!)` | ✗ | Actually n! grows faster |


---

## Key Difference

<div class="highlight">
<strong>Big-O vs Little-o:</strong><br>
O allows equality in growth rate, o does NOT.<br><br>
n = O(n) ✓    but    n = o(n) ✗<br>
n = O(n²) ✓   and    n = o(n²) ✓
</div>

---

# 6. Common Complexity Classes
### Hierarchy (Fastest to Slowest)

<div class="formula">
O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!) < O(nⁿ)
</div>

---

### Detailed Comparison Table

| Complexity | Name | n=10 | n=100 | n=1000 | Typical Examples |
|-----------|------|------|-------|--------|-----------------|
| O(1) | Constant | 1 | 1 | 1 | Array access, Hash lookup |
| O(log n) | Logarithmic | 3 | 7 | 10 | Binary search, BST ops |
| O(√n) | Square root | 3 | 10 | 32 | Prime checking, some optimizations |
| O(n) | Linear | 10 | 100 | 1000 | Linear search, single loop |
| O(n log n) | Linearithmic | 30 | 700 | 10K | Merge sort, Heap sort, Quick sort avg |
| O(n²) | Quadratic | 100 | 10K | 1M | Bubble sort, nested loops |
| O(n³) | Cubic | 1K | 1M | 1B | Floyd-Warshall, 3 nested loops |
| O(2ⁿ) | Exponential | 1K | ~10³⁰ | ∞ | Subset problems, brute force |
| O(n!) | Factorial | 3.6M | ∞ | ∞ | Permutations, TSP brute force |

---

# 6. Common Complexity Classes — Placement Insight

## What to Expect

<div class="important">
<strong>Target for coding interviews:</strong><br>
• O(1) or O(log n) → Excellent<br>
• O(n) or O(n log n) → Good/Acceptable<br>
• O(n²) → Acceptable for small n, try to optimize<br>
• O(2ⁿ) or O(n!) → Only for NP-hard problems with constraints
</div>

---

## The "1 Billion Operations Rule"

A standard computer can execute roughly 10⁹ (or 1 billion) basic operations in 1 second.

| Time Limit | Max n for O(n) | Max n for O(n²) | Max n for O(n log n) |
|-----------|---------------|-----------------|---------------------|
| 1 second | ~10⁸ | ~10⁴ | ~10⁷ |
| 2 seconds | ~2×10⁸ | ~1.4×10⁴ | ~2×10⁷ |

<div class="highlight">
<strong>Tip:</strong> In competitive programming, if n ≤ 10⁵, aim for O(n log n) or better.<br>
If n ≤ 10³, O(n²) might be acceptable.
</div>

---

# 7. Analyzing Time Complexity

## Step-by-Step Method

### 1. Identify the Basic Operation
- The operation executed most frequently (usually inside innermost loop)
- Count how many times it executes as a function of `n`

---

### 2. Analyze Loop Structures

```python
# Single loop → O(n)
for i in range(n):
    print(i)          # Runs n times

# Nested loops → O(n²)
for i in range(n):
    for j in range(n):
        print(i, j)   # Runs n × n = n² times

# Dependent nested loops → O(n²)
for i in range(n):
    for j in range(i, n):
        print(i, j)   # Runs n + (n-1) + ... + 1 = n(n+1)/2 = O(n²)
```

---

# 7. Analyzing Time Complexity (Cont.)

### 3. Analyze Loop with Different Increments

```python
# Loop divides by 2 → O(log n)
i = n
while i > 0:
    print(i)
    i = i // 2        # i = n, n/2, n/4, ... , 1 → log₂(n) iterations

# Loop with multiplication → O(log n)
i = 1
while i < n:
    print(i)
    i = i * 2         # i = 1, 2, 4, 8, ... → log₂(n) iterations

# Loop with square root → O(√n)
for i in range(1, int(n**0.5) + 1):
    print(i)          # Runs √n times
```

---

# 7. Analyzing Time Complexity (Cont.)

### 4. Multiple Independent Loops

```python
# Sequential loops → O(n) + O(n) = O(n)
for i in range(n):
    print(i)
for j in range(n):
    print(j)

# Different variables → O(n + m)
for i in range(n):
    print(i)
for j in range(m):
    print(j)

# Nested with different variables → O(n × m)
for i in range(n):
    for j in range(m):
        print(i, j)
```

<div class="highlight">
<strong>Rule:</strong> Sequential loops → ADD complexities && Nested loops → MULTIPLY complexities
</div>

---

# 7. Analyzing Time Complexity : Recursive Algorithms

## Recurrence Relations

```python
# Factorial: T(n) = T(n-1) + O(1) → O(n)
def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)

# Fibonacci (naive): T(n) = T(n-1) + T(n-2) + O(1) → O(2ⁿ)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

# Binary Search: T(n) = T(n/2) + O(1) → O(log n)
def binary_search(arr, target, low, high):
    if low > high: return -1
    mid = (low + high) // 2
    if arr[mid] == target: return mid
    elif arr[mid] > target: return binary_search(arr, target, low, mid-1)
    else: return binary_search(arr, target, mid+1, high)
```

---

# 8. Space Complexity Analysis

## What Counts as Space?

| Component | Counted? | Examples |
|-----------|----------|----------|
| Input space | Usually NOT counted | The input array itself |
| Auxiliary space | YES | Extra variables, recursion stack |
| Output space | Depends on problem | Result array |

---

## Examples

```python
# Space: O(1) auxiliary
def sum_array(arr):
    total = 0           # O(1)
    for x in arr:
        total += x
    return total

# Space: O(n) auxiliary
def reverse_array(arr):
    result = []         # O(n) for new array
    for i in range(len(arr)-1, -1, -1):
        result.append(arr[i])
    return result

# Space: O(n) due to recursion stack
def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)   # Stack depth = n
```

---

# 8. Space Complexity : Placement Tips

## Common Patterns

<div class="highlight">
<strong>Recursion Stack Depth:</strong><br>
• Linear recursion (factorial): O(n) stack space<br>
• Binary recursion (merge sort): O(log n) stack space<br>
• Tree recursion (naive fibonacci): O(n) stack space (worst case)
</div>


---

## In-Place vs Out-of-Place

| Algorithm | Time | Space | Type |
|-----------|------|-------|------|
| Bubble Sort | O(n²) | O(1) | In-place |
| Merge Sort | O(n log n) | O(n) | Out-of-place |
| Quick Sort | O(n log n) avg | O(log n) | In-place |
| Heap Sort | O(n log n) | O(1) | In-place |

<div class="important">
<strong>Interview Tip:</strong> If asked to optimize space, consider:<br>
• In-place algorithms (swap instead of creating new arrays)<br>
• Iterative instead of recursive (avoid stack overflow)<br>
• Bit manipulation for boolean flags
</div>

---

# 9. Recurrence Relations & Master Theorem

## Master Theorem

For recurrences of the form:

<div class="formula">
T(n) = a·T(n/b) + O(nᵈ)
</div>

Where `a ≥ 1`, `b > 1`, `d ≥ 0`

| Case | Condition | Solution |
|------|-----------|----------|
| 1 | a < bᵈ | T(n) = O(nᵈ) |
| 2 | a = bᵈ | T(n) = O(nᵈ · log n) |
| 3 | a > bᵈ | T(n) = O(n^(log_b a)) |

---

# 9. Master Theorem — Examples

## Example 1: Merge Sort
```
T(n) = 2·T(n/2) + O(n)
a = 2, b = 2, d = 1
a = bᵈ → 2 = 2¹ ✓ (Case 2)
T(n) = O(n¹ · log n) = O(n log n)
```

## Example 2: Binary Search
```
T(n) = T(n/2) + O(1)
a = 1, b = 2, d = 0
a = bᵈ → 1 = 2⁰ ✓ (Case 2)
T(n) = O(n⁰ · log n) = O(log n)
```

---

## Example 3: Strassen's Matrix Multiplication
```
T(n) = 7·T(n/2) + O(n²)
a = 7, b = 2, d = 2
a > bᵈ → 7 > 4 (Case 3)
T(n) = O(n^(log₂ 7)) ≈ O(n^2.81)
```

---

# 9. Master Theorem — More Examples

## Example 4: When Master Theorem Doesn't Apply
```
T(n) = 2·T(n/2) + O(n log n)
```
This doesn't fit the standard form (f(n) is not O(nᵈ)).

**Solution**: Use the **Extended Master Theorem** or **Recursion Tree Method**

## Example 5: Akra-Bazzi Method
For more general recurrences:
```
T(n) = Σ aᵢ·T(n/bᵢ) + f(n)
```

<div class="highlight">
<strong>Placement Tip:</strong> Master Theorem is asked frequently in interviews.<br>
Memorize the three cases and practice identifying a, b, and d quickly.
</div>

---

# 10. Amortized Analysis

## What is Amortized Analysis?

Average performance over a **sequence of operations**, even if some individual operations are expensive.

## Three Methods

### 1. Aggregate Analysis
- Total cost of n operations / n

### 2. Accounting Method
- Assign different charges to different operations
- Some operations are "overcharged" to pay for future expensive ones

---

### 3. Potential Method
- Define a potential function Φ
- Amortized cost = Actual cost + ΔΦ

---

# 10. Amortized Analysis — Examples

## Example: Dynamic Array (ArrayList)

```python
# Append operation on dynamic array
# Most appends: O(1)
# When full: resize to 2x capacity → O(n) to copy elements

# Aggregate Analysis:
# n appends: n operations of O(1) + log n resizes of O(n)
# Total: O(n) + O(n) = O(n)
# Amortized per operation: O(n)/n = O(1)
```

---

## Example: Stack with Multi-pop
```python
# push: O(1)
# pop: O(1)
# multi-pop(k): O(k) — can pop up to n elements

# n operations (push, pop, multi-pop):
# Each element pushed once and popped once
# Total: O(n) for n operations
# Amortized: O(1) per operation
```

<div class="highlight">
<strong>Key Insight:</strong> Amortized O(1) ≠ Worst-case O(1)<br>
Individual operations can be O(n), but sequence averages to O(1)
</div>

---

# 11. Placement-Ready Problem Solving

## Type 1: Find Complexity of Given Code

### Question: What is the time complexity?
```python
def mystery(n):
    count = 0
    for i in range(n):
        for j in range(i, n):
            count += 1
    return count
```

---

**Solution:**
- Inner loop runs: n + (n-1) + (n-2) + ... + 1 = n(n+1)/2
- Time: **O(n²)**

---

# 11. Placement-Ready Problem Solving (Cont.)

## Type 2: Find Complexity with Logarithmic Loop

### Question: What is the time complexity?
```python
def mystery(n):
    count = 0
    i = 1
    while i < n:
        for j in range(n):
            count += 1
        i = i * 2
    return count
```

---

**Solution:**
- Outer while: i = 1, 2, 4, 8, ... → log₂(n) iterations
- Inner for: n iterations each time
- Total: n × log(n) = **O(n log n)**

---

# 11. Placement-Ready Problem Solving (Cont.)

## Type 3: Nested Loops with Different Increments

### Question: What is the time complexity?
```python
def mystery(n):
    count = 0
    for i in range(n):
        j = 1
        while j < n:
            count += 1
            j = j * 2
    return count
```

---

**Solution:**
- Outer loop: n iterations
- Inner while: log₂(n) iterations
- Total: n × log(n) = **O(n log n)**

---

# 11. Placement-Ready Problem Solving (Cont.)

## Type 4: Multiple Variables

### Question: What is the time complexity?
```python
def mystery(n, m):
    count = 0
    for i in range(n):
        for j in range(m):
            count += 1
    return count
```

---

**Solution:**
- Total: n × m = **O(n·m)**
- If n = m: **O(n²)**

---

# 11. Placement-Ready Problem Solving (Cont.)

## Type 5: Recursion with Multiple Calls

### Question: What is the time complexity?
```python
def mystery(n):
    if n <= 1:
        return 1
    return mystery(n-1) + mystery(n-2)
```

---

**Solution:**
- Recurrence: T(n) = T(n-1) + T(n-2) + O(1)
- Similar to Fibonacci
- Time: **O(2ⁿ)** (exponential)

### Optimized with Memoization:
```python
def mystery(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return 1
    memo[n] = mystery(n-1) + mystery(n-2)
    return memo[n]
```
- Time: **O(n)**, Space: **O(n)**

---

# 11. Placement-Ready Problem Solving (Cont.)

## Type 6: Divide and Conquer

### Question: What is the time complexity?
```python
def mystery(arr, low, high):
    if low >= high:
        return
    mid = (low + high) // 2
    mystery(arr, low, mid)
    mystery(arr, mid + 1, high)
    # Merge step: O(n)
    merge(arr, low, mid, high)
```

---

**Solution:**
- Recurrence: T(n) = 2T(n/2) + O(n)
- a = 2, b = 2, d = 1 → Case 2 of Master Theorem
- Time: **O(n log n)**

---

# 11. Placement-Ready Problem Solving (Cont.)

## Type 7: Space Complexity Analysis

### Question: What is the auxiliary space complexity?
```python
def mystery(n):
    if n <= 0:
        return
    print(n)
    mystery(n - 1)
    mystery(n - 1)
```

---

**Solution:**
- Recursion tree has height n
- But it's a binary tree, so max stack depth at any point is n
- Space: **O(n)** (stack depth)

---

# 12. Quick Reference Cheat Sheet

## Notation Summary

| Notation | Name | Meaning | Use Case |
|----------|------|---------|----------|
| O | Big-O | Upper bound | Worst-case analysis |
| Ω | Big-Omega | Lower bound | Best-case analysis |
| Θ | Big-Theta | Tight bound | Exact growth rate |
| o | Little-o | Strictly smaller | Theoretical proofs |
| ω | Little-omega | Strictly larger | Theoretical proofs |

## Complexity Ordering
```
O(1) ⊂ O(log n) ⊂ O(√n) ⊂ O(n) ⊂ O(n log n) ⊂ O(n²) ⊂ O(n³) ⊂ O(2ⁿ) ⊂ O(n!) ⊂ O(nⁿ)
```

---

# 12. Quick Reference — Master Theorem

```
T(n) = a·T(n/b) + O(nᵈ)

Case 1: a < bᵈ   → T(n) = O(nᵈ)
Case 2: a = bᵈ   → T(n) = O(nᵈ · log n)
Case 3: a > bᵈ   → T(n) = O(n^(log_b a))
```

---

## Common Algorithm Complexities

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Linear Search | O(1) | O(n) | O(n) | O(1) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) |

---

# 12. Quick Reference — Interview Checklist

## Before the Interview

<div class="highlight">
✅ Memorize complexity of all standard sorting algorithms<br>
✅ Practice deriving complexity from code snippets<br>
✅ Know when to use O vs Θ vs Ω<br>
✅ Understand amortized analysis basics<br>
✅ Be able to explain Master Theorem with examples
</div>

---

## During the Interview

<div class="important">
1. <strong>Always state complexity clearly</strong> - "This is O(n log n) time and O(1) space"

2. <strong>Distinguish best/average/worst case</strong> when relevant
3. <strong>Mention if you can optimize</strong> - "Can we do better than O(n²)?"
4. <strong>Explain your reasoning</strong> - don't just state the answer
5. <strong>Be careful with recursive space</strong> - include stack space
</div>


---

## Common Interview Questions
- "What's the time/space complexity of your solution?"
- "Can you optimize this further?"
- "What's the best possible complexity for this problem?"
- "Explain why this is O(n log n)"

---

<!-- _class: lead -->
# Thank You!
## Best of Luck for Your Placements PREP! 🎯

**Key Takeaways:**
- Master Big-O, Big-Omega, Big-Theta
- Practice analyzing code complexity
- Know standard algorithm complexities by heart
- Understand space vs time tradeoffs

**Keep Practicing!**
