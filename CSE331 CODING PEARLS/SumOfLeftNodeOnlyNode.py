class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def sum_of_left_only_nodes(root):
    # Base case: If the tree is empty, return 0
    if root is None:
        return 0
    
    # Initialize the sum
    left_sum = 0
    
    # Check if the current node has a left child with no right sibling
    if root.left is not None and root.right is None:
        #left_sum += root.left.value
        left_sum += root.value
    
    # Recur for the left and right subtrees
    left_sum += sum_of_left_only_nodes(root.left)
    left_sum += sum_of_left_only_nodes(root.right)
    
    # Return the computed sum
    return left_sum

# Example usage:
# Construct the binary tree:
#         10
#        /  \
#       5    15
#      /      \
#     3        20

root = TreeNode(10)
root.left = TreeNode(5)
root.right = TreeNode(15)
root.left.left = TreeNode(3)
root.right.right = TreeNode(20)

# Compute the sum of nodes that have only left children
result = sum_of_left_only_nodes(root)
print("Sum of nodes with only left child:", result)  # Output should be 5
