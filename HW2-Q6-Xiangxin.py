import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from community import community_louvain
import seaborn as sns
from collections import Counter
import matplotlib.colors as mcolors

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AirTrafficAnalyzer:
    def __init__(self, master_file, traffic_file):
        """Initialize the analyzer with data files"""
        self.master_file = master_file
        self.traffic_file = traffic_file
        self.G = None
        self.airport_info = None
        self.load_data()

    def load_data(self):
        """Load and preprocess the air traffic data"""
        print("Loading airport master data...")
        self.airport_info = pd.read_csv(self.master_file)

        print("Loading traffic data...")
        traffic_data = pd.read_csv(self.traffic_file)

        # Create directed graph from traffic data
        self.G = nx.DiGraph()

        # Add edges with weight (number of flights per month)
        flight_counts = traffic_data.groupby(['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']).size().reset_index(
            name='flights')

        for _, row in flight_counts.iterrows():
            origin = row['ORIGIN_AIRPORT_ID']
            dest = row['DEST_AIRPORT_ID']
            flights = row['flights']

            # Add nodes with airport information if available
            if origin not in self.G:
                origin_info = self.airport_info[self.airport_info['AIRPORT_ID'] == origin]
                if not origin_info.empty:
                    self.G.add_node(origin, **origin_info.iloc[0].to_dict())
                else:
                    self.G.add_node(origin)

            if dest not in self.G:
                dest_info = self.airport_info[self.airport_info['AIRPORT_ID'] == dest]
                if not dest_info.empty:
                    self.G.add_node(dest, **dest_info.iloc[0].to_dict())
                else:
                    self.G.add_node(dest)

            # Add edge with weight
            self.G.add_edge(origin, dest, weight=flights)

        print(f"Graph created with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")

    def analyze_communities(self):
        """6a: Analyze communities in the air traffic network"""
        print("\n" + "=" * 60)
        print("6a: COMMUNITY ANALYSIS")
        print("=" * 60)

        # Convert to undirected graph for community detection
        G_undirected = self.G.to_undirected()

        # Remove isolated nodes for community detection
        G_connected = G_undirected.subgraph(max(nx.connected_components(G_undirected), key=len))

        # Detect communities using Louvain method
        partition = community_louvain.best_partition(G_connected)

        # Analyze community structure
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

        # Sort communities by size
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)

        print(f"Number of communities detected: {len(communities)}")
        print("\nTop 10 largest communities:")
        for i, (comm_id, nodes) in enumerate(sorted_communities[:10]):
            print(f"Community {comm_id}: {len(nodes)} airports")

            # Get major airports in this community
            degrees = [(node, self.G.degree(node)) for node in nodes if node in self.G]
            if degrees:
                top_airports = sorted(degrees, key=lambda x: x[1], reverse=True)[:3]
                airport_names = []
                for airport_id, deg in top_airports:
                    info = self.airport_info[self.airport_info['AIRPORT_ID'] == airport_id]
                    if not info.empty:
                        name = info['DISPLAY_AIRPORT_NAME'].iloc[0]
                        state = info['AIRPORT_STATE_CODE'].iloc[0]
                        airport_names.append(f"{name} ({state})")
                print(f"  Major hubs: {', '.join(airport_names)}")

        # Visualize community sizes
        community_sizes = [len(nodes) for _, nodes in sorted_communities]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(len(community_sizes)), community_sizes)
        plt.xlabel('Community ID')
        plt.ylabel('Number of Airports')
        plt.title('Community Size Distribution')
        plt.xticks(range(len(community_sizes)), [f'C{i}' for i in range(len(community_sizes))])

        plt.subplot(1, 2, 2)
        # Show only top 10 communities for pie chart clarity
        top_10_sizes = community_sizes[:10]
        top_10_labels = [f'Community {i}' for i in range(10)]
        if len(community_sizes) > 10:
            top_10_sizes.append(sum(community_sizes[10:]))
            top_10_labels.append('Other Communities')

        plt.pie(top_10_sizes, labels=top_10_labels, autopct='%1.1f%%')
        plt.title('Community Distribution (Top 10)')

        plt.tight_layout()
        plt.show()

        return partition

    def analyze_pagerank(self):
        """6b: Compute and analyze PageRank values"""
        print("\n" + "=" * 60)
        print("6b: PAGERANK ANALYSIS")
        print("=" * 60)

        # Compute PageRank
        pagerank = nx.pagerank(self.G, alpha=0.85, weight='weight')

        # Get top airports by PageRank
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]

        print("Top 15 airports by PageRank:")
        for i, (airport_id, score) in enumerate(top_pagerank, 1):
            info = self.airport_info[self.airport_info['AIRPORT_ID'] == airport_id]
            if not info.empty:
                name = info['DISPLAY_AIRPORT_NAME'].iloc[0]
                state = info['AIRPORT_STATE_CODE'].iloc[0]
                print(f"{i:2d}. {name} ({state}): {score:.6f}")
            else:
                print(f"{i:2d}. Airport ID {airport_id}: {score:.6f}")

        # Plot PageRank distribution
        pr_values = list(pagerank.values())

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(pr_values, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('PageRank Value')
        plt.ylabel('Frequency')
        plt.title('PageRank Distribution')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        # Log-log plot to check for power law
        sorted_pr = np.sort(pr_values)[::-1]
        ranks = np.arange(1, len(sorted_pr) + 1)
        plt.loglog(ranks, sorted_pr, 'o-', alpha=0.7)
        plt.xlabel('Rank')
        plt.ylabel('PageRank Value')
        plt.title('Rank vs PageRank (Log-Log)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Statistics
        print(f"\nPageRank Statistics:")
        print(f"Mean: {np.mean(pr_values):.6f}")
        print(f"Std: {np.std(pr_values):.6f}")
        print(f"Max: {max(pr_values):.6f}")
        print(f"Min: {min(pr_values):.6f}")

        return pagerank

    def analyze_hits(self):
        """6c: Compute and analyze Hub and Authority scores"""
        print("\n" + "=" * 60)
        print("6c: HUB AND AUTHORITY ANALYSIS")
        print("=" * 60)

        # Compute HITS scores
        hubs, authorities = nx.hits(self.G, max_iter=1000, normalized=True)

        # Get top hubs
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_auth = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:10]

        print("Top 10 Hub airports (important for outgoing flights):")
        for i, (airport_id, score) in enumerate(top_hubs, 1):
            info = self.airport_info[self.airport_info['AIRPORT_ID'] == airport_id]
            if not info.empty:
                name = info['DISPLAY_AIRPORT_NAME'].iloc[0]
                state = info['AIRPORT_STATE_CODE'].iloc[0]
                print(f"{i:2d}. {name} ({state}): {score:.6f}")

        print("\nTop 10 Authority airports (important for incoming flights):")
        for i, (airport_id, score) in enumerate(top_auth, 1):
            info = self.airport_info[self.airport_info['AIRPORT_ID'] == airport_id]
            if not info.empty:
                name = info['DISPLAY_AIRPORT_NAME'].iloc[0]
                state = info['AIRPORT_STATE_CODE'].iloc[0]
                print(f"{i:2d}. {name} ({state}): {score:.6f}")

        # Plot distributions
        hub_values = list(hubs.values())
        auth_values = list(authorities.values())

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(hub_values, bins=50, alpha=0.7, label='Hubs', edgecolor='black')
        plt.xlabel('Hub Score')
        plt.ylabel('Frequency')
        plt.title('Hub Score Distribution')
        plt.yscale('log')

        plt.subplot(1, 3, 2)
        plt.hist(auth_values, bins=50, alpha=0.7, label='Authorities', color='orange', edgecolor='black')
        plt.xlabel('Authority Score')
        plt.ylabel('Frequency')
        plt.title('Authority Score Distribution')
        plt.yscale('log')

        plt.subplot(1, 3, 3)
        # Scatter plot of hub vs authority scores
        airport_ids = list(hubs.keys())
        hub_scores = [hubs[aid] for aid in airport_ids]
        auth_scores = [authorities[aid] for aid in airport_ids]

        plt.scatter(hub_scores, auth_scores, alpha=0.6)
        plt.xlabel('Hub Score')
        plt.ylabel('Authority Score')
        plt.title('Hub vs Authority Scores')
        plt.xscale('log')
        plt.yscale('log')

        # Add identity line
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'k--', alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Statistics
        print(f"\nHub Score Statistics:")
        print(f"Mean: {np.mean(hub_values):.6f}")
        print(f"Std: {np.std(hub_values):.6f}")

        print(f"\nAuthority Score Statistics:")
        print(f"Mean: {np.mean(auth_values):.6f}")
        print(f"Std: {np.std(auth_values):.6f}")

        return hubs, authorities

    def comprehensive_analysis(self):
        """Run complete analysis for question 6"""
        print("US DOMESTIC AIR TRAFFIC NETWORK ANALYSIS")
        print("=" * 60)

        # Basic network statistics
        print("Network Overview:")
        print(f"Number of airports: {self.G.number_of_nodes()}")
        print(f"Number of routes: {self.G.number_of_edges()}")
        print(f"Network density: {nx.density(self.G):.6f}")
        print(f"Average degree: {sum(dict(self.G.degree()).values()) / self.G.number_of_nodes():.2f}")

        # Degree distribution analysis
        degrees = [d for n, d in self.G.degree()]
        print(f"Maximum degree: {max(degrees)}")
        print(f"Minimum degree: {min(degrees)}")

        # Run all analyses
        communities = self.analyze_communities()
        pagerank = self.analyze_pagerank()
        hubs, authorities = self.analyze_hits()

        return communities, pagerank, hubs, authorities


# Main execution
if __name__ == "__main__":
    # Initialize analyzer with the provided files
    analyzer = AirTrafficAnalyzer(
        master_file="288804893_T_MASTER_CORD.csv",
        traffic_file="288798530_T_T100D_MARKET_ALL_CARRIER.csv"
    )

    # Run complete analysis
    communities, pagerank, hubs, authorities = analyzer.comprehensive_analysis()

    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    print("""
    Key Insights:

    1. COMMUNITY STRUCTURE:
    - The US air traffic network shows clear geographical clustering
    - Communities typically correspond to regional hubs and their surrounding airports
    - Major hubs like Atlanta, Chicago, and Dallas form the centers of large communities

    2. PAGERANK ANALYSIS:
    - PageRank identifies the most centrally important airports in the network
    - These are typically major international hubs with extensive connectivity
    - The distribution shows a heavy-tailed pattern, characteristic of scale-free networks

    3. HUB & AUTHORITY ANALYSIS:
    - Hub scores identify airports that are important for distributing traffic
    - Authority scores identify airports that are popular destinations
    - Some airports serve both roles (high hub and authority scores)
    - The correlation between hub and authority scores reveals airport roles

    The analysis reveals a hierarchical structure with major hubs connecting regional clusters,
    which is efficient for national air transportation but creates vulnerability to hub disruptions.
    """)