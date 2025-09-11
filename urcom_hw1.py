import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load flight data and airport details
flights = pd.read_csv("288798530_T_T100D_MARKET_ALL_CARRIER.csv")
airports = pd.read_csv("288804893_T_MASTER_CORD.csv")

# Create graph
G = nx.Graph()
for _, row in flights.iterrows():
    G.add_edge(row['ORIGIN_AIRPORT_ID'], row['DEST_AIRPORT_ID'])

components = list(nx.connected_components(G))
print(f"Number of connected components: {len(components)}")
print(f"Size of largest component: {len(max(components, key=len))}")

clustering = nx.clustering(G)
plt.hist(clustering.values(), bins=50)
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.show()

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Plot degree centrality distribution
plt.hist(degree_centrality.values(), bins=50)
plt.xlabel("Degree Centrality")
plt.ylabel("Frequency")
plt.show()

# Plot betweenness centrality distribution
plt.hist(betweenness_centrality.values(), bins=50)
plt.xlabel("Betweenness Centrality")
plt.ylabel("Frequency")
plt.show()