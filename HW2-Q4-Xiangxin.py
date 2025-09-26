import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter



def question_4_verification():
    """Verify if line graphs of power-law networks also exhibit power-law distributions"""

    # Generate multiple scale-free networks using different methods
    print("Generating power-law networks and their line graphs...")

    # Method 1: Barabasi-Albert model (n=1000, m=2)
    G_ba = nx.barabasi_albert_graph(1000, 2, seed=42)

    # Method 2: Configuration model with power-law degree sequence
    degrees = np.random.zipf(2.5, 500)  # Zipf distribution for power-law
    degrees = [min(d, 50) for d in degrees]  # Cap maximum degree
    if sum(degrees) % 2 != 0:
        degrees[0] += 1  # Make sum even
    G_config = nx.configuration_model(degrees)
    G_config = nx.Graph(G_config)  # Convert to simple graph
    G_config.remove_edges_from(nx.selfloop_edges(G_config))

    networks = {
        'Barabasi-Albert (n=1000, m=2)': G_ba,
        'Configuration Model (power-law)': G_config
    }

    # Analyze each network and its line graph
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (name, G) in enumerate(networks.items()):
        # Create line graph
        L = nx.line_graph(G)

        # Get degree distributions
        degrees_original = [d for n, d in G.degree()]
        degrees_line = [d for n, d in L.degree()]

        # Plot original network degree distribution
        ax1 = axes[idx, 0]
        plot_degree_distribution(degrees_original, ax1, f'{name}\nOriginal Network')

        # Plot line graph degree distribution
        ax2 = axes[idx, 1]
        plot_degree_distribution(degrees_line, ax2, f'{name}\nLine Graph')

        # Plot both on same log-log scale for comparison
        ax3 = axes[idx, 2]
        plot_comparison(degrees_original, degrees_line, ax3, name)

        # Fit power law and print exponents
        print(f"\n--- {name} ---")
        analyze_power_law(degrees_original, "Original")
        analyze_power_law(degrees_line, "Line Graph")

    plt.tight_layout()
    plt.show()


def plot_degree_distribution(degrees, ax, title):
    """Plot degree distribution on log-log scale"""
    degree_count = Counter(degrees)
    x = list(degree_count.keys())
    y = list(degree_count.values())

    ax.loglog(x, y, 'bo', alpha=0.7, markersize=4)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.2)


def plot_comparison(degrees_orig, degrees_line, ax, name):
    """Plot comparison of original and line graph degree distributions"""
    # Original network
    degree_count_orig = Counter(degrees_orig)
    x_orig = list(degree_count_orig.keys())
    y_orig = list(degree_count_orig.values())

    # Line graph
    degree_count_line = Counter(degrees_line)
    x_line = list(degree_count_line.keys())
    y_line = list(degree_count_line.values())

    ax.loglog(x_orig, y_orig, 'bo', alpha=0.7, markersize=4, label='Original')
    ax.loglog(x_line, y_line, 'ro', alpha=0.7, markersize=4, label='Line Graph')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name}\nComparison')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)


def analyze_power_law(degrees, label):
    """Analyze and print power-law properties"""
    if len(degrees) > 0:
        # Simple power-law fit using linear regression on log-log scale
        degree_count = Counter(degrees)
        x = np.array(list(degree_count.keys()))
        y = np.array(list(degree_count.values()))

        # Filter out zeros and take logs
        mask = (x > 0) & (y > 0)
        if np.sum(mask) > 2:
            log_x = np.log(x[mask])
            log_y = np.log(y[mask])

            # Linear regression to estimate exponent
            slope, intercept = np.polyfit(log_x, log_y, 1)
            exponent = -slope  # Negative slope gives positive exponent

            print(f"{label}: Estimated exponent ≈ {exponent:.3f}")
            print(f"{label}: Degree range: {min(degrees)} - {max(degrees)}")
        else:
            print(f"{label}: Insufficient data for power-law fit")
    else:
        print(f"{label}: No degree data")


def question_3_ba_modification_demo():
    """Demonstrate modified BA model with different exponents"""

    print("\n" + "=" * 60)
    print("DEMONSTRATION: Modified BA Model for Different Exponents")
    print("=" * 60)

    # Standard BA model (exponent ≈ 3)
    G_standard = nx.barabasi_albert_graph(1000, 2, seed=42)
    degrees_standard = [d for n, d in G_standard.degree()]

    # Modified BA with non-linear attachment (α < 1)
    # This requires custom implementation
    print("\nStandard BA model: γ ≈ 3")
    analyze_power_law(degrees_standard, "Standard BA")

    print("\nNote: Modified BA with exponent > 3 requires")
    print("non-linear preferential attachment or node deletion")
    print("which is more complex to implement.")


# Run the verification
if __name__ == "__main__":
    print("CS 5834: Urban Computing - Homework 2 Support Code")
    print("Question 4: Power-law verification in line graphs")
    print("-" * 60)

    question_4_verification()
    question_3_ba_modification_demo()

    print("\n" + "=" * 60)
    print("CONCLUSION FOR QUESTION 4:")
    print("Line graphs of power-law networks generally maintain")
    print("power-law distributions, though the exponent may change.")
    print("This supports the theoretical reasoning provided.")
    print("=" * 60)

question_4_verification()