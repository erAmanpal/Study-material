# Install first: pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Sample data
transactions = [
    ['Apple', 'Bear', 'Rice','Chicken'],
    ['Apple', 'Bear', 'Rice'],
    ['Apple', 'Bear'],
    ['Apple', 'Pear'],
    ['Milk', 'Bear', 'Rice','Chicken'],
    ['Milk', 'Bear', 'Rice'],
    ['Milk', 'Bear'],
    ['Milk', 'Pear'],
]

# Step 1: Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Step 2: Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print(frequent_itemsets)
# Step 3: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Show results
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Step 4: Visualize rules as a graph
import plotly.graph_objects as go

# Prepare data
edges = []
for _, row in rules.iterrows():
    antecedent = ', '.join(row['antecedents'])
    consequent = ', '.join(row['consequents'])
    confidence = row['confidence']
    lift = row['lift']
    label = f"Conf: {confidence:.2f}, Lift: {lift:.2f}"
    
    # Edge length inversely proportional to confidence
    length = 1 / confidence if confidence > 0 else 1
    edges.append((antecedent, consequent, label, lift, length))

# Extract unique nodes
nodes = list(set([e[0] for e in edges] + [e[1] for e in edges]))
node_indices = {node: i for i, node in enumerate(nodes)}

# Create edge traces
edge_traces = []
for e in edges:
    x0 = node_indices[e[0]]
    x1 = node_indices[e[1]]
    y0 = 1
    y1 = 1 + e[4]  # edge length based on confidence
    edge_traces.append(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode='lines+text',
        line=dict(width=e[3]*2, color='gray'),  # thickness from lift
        text=[e[2]],
        hoverinfo='text',
        showlegend=False
    ))

# Create node trace
node_trace = go.Scatter(
    x=list(node_indices.values()),
    y=[1]*len(node_indices),
    mode='markers+text',
    marker=dict(size=20, color='lightblue'),
    text=list(node_indices.keys()),
    textposition='top center',
    hoverinfo='text'
)

# Combine and plot
fig = go.Figure(data=edge_traces + [node_trace])
fig.update_layout(title='Association Rule Graph (Lift = Thickness, Confidence = Length)', showlegend=False)
fig.show()
