import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def run_firefly_algorithm():
    """Runs the Firefly Algorithm (Rust executable) and waits for completion."""
    subprocess.run(["cargo", "run"], check=True)

def load_firefly_data(file_path):
    """Loads Firefly Algorithm results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data["mesh_routers"]), np.array(data["mesh_clients"]), data["best_fitness"], data["sgc"], data["ncmc"], data["ncmcpr"]

def plot_mesh_network(routers, clients, coverage_radius=4.5, comm_radius=4.5, area_size=32, output_file='firefly_plot.png'):
    """Plots the mesh network result."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    
    # Plot coverage areas (pink circles)
    for router in routers:
        circle = plt.Circle((router[0], router[1]), coverage_radius, color='pink', alpha=0.4)
        ax.add_patch(circle)
    
    # Plot mesh routers (blue points)
    ax.scatter(routers[:, 0], routers[:, 1], color='blue', label='Mesh Routers', zorder=3)
    
    # Plot mesh clients (green points)
    ax.scatter(clients[:, 0], clients[:, 1], color='green', label='Mesh Clients', zorder=2)
    
    # Draw connections between routers within communication radius
    for i in range(len(routers)):
        for j in range(i + 1, len(routers)):
            dist = np.linalg.norm(routers[i] - routers[j])
            if dist <= comm_radius:
                ax.plot([routers[i, 0], routers[j, 0]], [routers[i, 1], routers[j, 1]], 'k-', linewidth=0.8)
    
    # Labels and legend
    ax.legend()
    ax.set_title("Best Mesh Network Solution Found by Firefly Algorithm")
    plt.savefig(output_file, dpi=300)
    plt.show()

# Step 1: Run Firefly Algorithm (Rust program)
run_firefly_algorithm()

# Step 2: Load results from the Firefly Algorithm output file
routers, clients, best_fitness, sgc, ncmc, ncmcpr = load_firefly_data("firefly_results.json")

# Step 3: Print the best fitness score and other key metrics
print(f"Best Fitness Score: {best_fitness}")
print(f"Size of Giant Component (SGC): {sgc}")
print(f"Number of Covered Mesh Clients (NCMC): {ncmc}")
print(f"Number of Covered Mesh Clients per Router (NCMCpR): {ncmcpr}")

# Step 4: Plot the mesh network
plot_mesh_network(routers, clients, output_file='firefly_plot.png')
