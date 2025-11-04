import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt

# Sample transactions
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Butter', 'Jam'],
    ['Bread', 'Jam']
]

# Step 1: Encode transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
# Step 2: Generate frequent itemsets and rules
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# warning is a common one when using mlxtend's association_rules() function.
# for some rules, the denominator (certainty_denom) is zero,
# which causes a division by zero — resulting in NaN or inf
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Step 3: Build graph
G = nx.DiGraph()

for _, row in rules.iterrows():
    antecedent = ', '.join(list(row['antecedents']))
    consequent = ', '.join(list(row['consequents']))
    confidence = row['confidence']
    lift = row['lift']
    
    G.add_edge(antecedent, consequent, weight=confidence, lift=lift)
# Step 4: Draw graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=12)

# Draw edges with confidence as thickness
nx.draw_networkx_edges(G, pos, edgelist=edges,
                       width=[d['weight'] * 5 for (_, _, d) in edges],
                       edge_color='gray', arrows=True)

# Add edge labels (lift)
edge_labels = {(u, v): f"Lift: {d['lift']:.2f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')

plt.title("Association Rules Network")
plt.axis('off')
plt.show()

