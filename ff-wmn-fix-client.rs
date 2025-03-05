use rand::Rng;
use std::collections::VecDeque;

const number_of_mesh_routers: usize = 16; // Number of mesh routers
const number_of_mesh_clients: usize = 32; // Number of mesh clients
const dimensions: usize = 2;  // Dimension of the problem
const number_of_iterations: usize = 100; // Number of iterations
const Alpha: f64 = 0.5; // Random parameter
const Beta0: f64 = 1.0; // Attractive constant
const Gamma: f64 = 1.0; // Light absorption coefficient
const LOWER_BOUND: f64 = 0.0;
const UPPER_BOUND: f64 = 32.0;
const minimum_communication_distance: f64 = 3.0; // Minimum signal coverage radius
const maximum_communication_distance: f64 = 4.5; // Maximum signal coverage radius

// User-defined weights for fitness components
const priority_of_SGC: f64 = 0.8;  // Weight for Size of Giant Component
const priority_of_NCMC: f64 = 0.1; // Weight for Number of Covered Mesh Clients
const priority_of_NCMCpR: f64 = 0.1; // Weight for Number of Covered Mesh Clients per Router

// Function to compute Size of Giant Component (SGC)
fn sgc(routers: &[[f64; dimensions]]) -> f64 {
    let mut largest_component = 0;
    let mut visited = vec![false; routers.len()];
    
    for start in 0..routers.len() {
        if !visited[start] {
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            let mut component_size = 1;
            
            while let Some(current) = queue.pop_front() {
                for (i, other_router) in routers.iter().enumerate() {
                    if !visited[i] {
                        let dist = distance(&routers[current], other_router);
                        if dist <= maximum_communication_distance {
                            visited[i] = true;
                            queue.push_back(i);
                            component_size += 1;
                    }
                    }
                }
            }
            largest_component = largest_component.max(component_size);
        }
    }
    (largest_component as f64 / number_of_mesh_routers as f64) * 100.0
}

// Function to compute Number of Covered Mesh Clients (NCMC)
fn ncmc(routers: &[[f64; dimensions]], clients: &[[f64; dimensions]]) -> f64 {
    let mut covered_clients = 0;
    
    for client in clients {
        for router in routers {
            let dist = distance(router, client);
            if dist <= maximum_communication_distance {
                covered_clients += 1;
                break;
            }
        }
    }
    
    (covered_clients as f64 / number_of_mesh_clients as f64) * 100.0
}

// Function to compute Number of Covered Mesh Clients per Router (NCMCpR)
fn ncmcpr(routers: &[[f64; dimensions]], clients: &[[f64; dimensions]]) -> f64 {
    let ncmc = ncmc(routers, clients);
    ncmc as f64 / routers.len() as f64 // Average number of clients per router
}

// Combined fitness function with user-defined weights
fn fitness_function(routers: &[[f64; dimensions]], clients: &[[f64; dimensions]]) -> f64 {
    let sgc = sgc(routers) as f64;
    let ncmc = ncmc(routers, clients) as f64;
    let ncmcpr = ncmcpr(routers, clients);
    
    // User-defined weighted sum of all three factors
    (priority_of_SGC * sgc) + (priority_of_NCMC * ncmc) + (priority_of_NCMCpR * ncmcpr)
}

// Cartesian Distance between two points
fn distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| (xi - yi).powi(2)).sum::<f64>().sqrt()
}

use std::fs::File;
use std::io::Write;
use serde_json::json;

fn save_results(routers: &Vec<[f64; dimensions]>, clients: &Vec<[f64; dimensions]>, best_fitness: f64, sgc: usize, ncmc: usize, ncmcpr: f64) {
    let data = json!({
        "mesh_routers": routers,
        "mesh_clients": clients,
        "best_fitness": best_fitness,
        "sgc": sgc,
        "ncmc": ncmc,
        "ncmcpr": ncmcpr
    });

    let mut file = File::create("firefly_results.json").expect("Unable to create file");
    file.write_all(data.to_string().as_bytes()).expect("Unable to write data");
}

fn firefly_algorithm() {
    let mut rng = rand::thread_rng();
    let mut mesh_routers = vec![[0.0; dimensions]; number_of_mesh_routers];
    let mut mesh_clients = vec![[0.0; dimensions]; number_of_mesh_clients];

    // Initialize mesh_clients randomly
    for i in 0..number_of_mesh_clients {
        for j in 0..dimensions {
            mesh_clients[i][j] = rng.gen_range(LOWER_BOUND..UPPER_BOUND);
        }
    }
    
    // Initialize mesh_routers randomly
    for i in 0..number_of_mesh_routers {
        for j in 0..dimensions {
            mesh_routers[i][j] = rng.gen_range(LOWER_BOUND..UPPER_BOUND);
        }
    }
    
    let mut best_mesh_routers = mesh_routers.clone();
    let mut best_fitness = 0.0;

    // Firefly Algorithm Iterations
    for _ in 0..number_of_iterations {
        for i in 0..number_of_mesh_routers {
            for j in 0..number_of_mesh_routers {
                if i != j { // Fireflies only compare to others, not themselves
                    let r_ij = distance(&mesh_routers[i], &mesh_routers[j]);
                    let beta = Beta0 * (-Gamma * r_ij * r_ij).exp();

                    for d in 0..dimensions {
                        let attraction = beta * (mesh_routers[j][d] - mesh_routers[i][d]);
                        let randomness = Alpha * (rng.gen::<f64>() - 0.5);

                        // Update firefly position based on the equation
                        mesh_routers[i][d] += attraction + randomness;
                        mesh_routers[i][d] = mesh_routers[i][d].clamp(LOWER_BOUND, UPPER_BOUND);
                    }
                }
            }
        }

        // Check and update best fitness
        let current_fitness = fitness_function(&mesh_routers, &mesh_clients);
        if current_fitness > best_fitness {
            best_fitness = current_fitness;
            best_mesh_routers = mesh_routers.clone();
        }
    }

    // Save results to JSON file
    let sgc_value = sgc(&best_mesh_routers);
    let ncmc_value = ncmc(&best_mesh_routers, &mesh_clients);
    let ncmcpr_value = ncmcpr(&best_mesh_routers, &mesh_clients);
    save_results(&best_mesh_routers, &mesh_clients, best_fitness, sgc_value, ncmc_value, ncmcpr_value);
    
    // Print results
    println!("Final Fitness Score: {}", best_fitness);
    println!("Results saved to firefly_results.json");
    println!("Best solution: {:?}", best_mesh_routers);
    println!("Mesh clients located at: {:?}", mesh_clients);
}
    }
    
    // Initialize mesh_routers randomly and calculate initial fitness
    for i in 0..number_of_mesh_routers {
    for j in 0..number_of_mesh_routers {
        let r_ij = distance(&mesh_routers[i], &mesh_routers[j]);
        let beta = Beta0 * (-Gamma * r_ij * r_ij).exp(); // β0 * exp(-γ * r_ij²)

                            for d in 0..dimensions {
            let attraction = beta * (mesh_routers[j][d] - mesh_routers[i][d]); // β(x_j - x_i)
            let randomness = Alpha * (rng.gen::<f64>() - 0.5); // α * ε

            // Update position based on the equation
            mesh_routers[i][d] += attraction + randomness;
            mesh_routers[i][d] = mesh_routers[i][d].clamp(LOWER_BOUND, UPPER_BOUND);
        }
        } // End of dimension loop
    } // End of second for loop
} // End of second for loop
    } // End of first for loop
    
    
    let fitness_score = fitness_function(&mesh_routers, &mesh_clients);
    println!("Initial Fitness Score: {}", fitness_score);

    let mut best_mesh_routers = mesh_routers.clone();
let mut best_fitness = fitness_function(&mesh_routers, &mesh_clients);

for _ in 0..number_of_iterations {
    for i in 0..number_of_mesh_routers {
        for j in 0..number_of_mesh_routers {
            let r = distance(&mesh_routers[i], &mesh_routers[j]);
            let beta = Beta0 * (-Gamma * r * r).exp();
            for d in 0..dimensions {
                mesh_routers[i][d] += beta * (mesh_routers[j][d] - mesh_routers[i][d]) +
                    Alpha * (rng.gen::<f64>() - 0.5);
                mesh_routers[i][d] = mesh_routers[i][d].clamp(LOWER_BOUND, UPPER_BOUND);
            }
        }
    }
    
    let current_fitness = fitness_function(&mesh_routers, &mesh_clients);
    if current_fitness > best_fitness {
        best_fitness = current_fitness;
        best_mesh_routers = mesh_routers.clone();
    }
}
            }
        }
    }

    // Print best solution found
    let final_fitness = fitness_function(&mesh_routers, &mesh_clients);
    println!("Final Fitness Score: {}", final_fitness);
    
    // Save results to JSON file
    let sgc_value = sgc(&mesh_routers);
    let ncmc_value = ncmc(&mesh_routers, &mesh_clients);
    let ncmcpr_value = ncmcpr(&mesh_routers, &mesh_clients);
    save_results(&mesh_routers, &mesh_clients, best_fitness, sgc_value, ncmc_value, ncmcpr_value);
    println!("Results saved to firefly_results.json");
    println!("Best solution: {:?}", mesh_routers);
    println!("Mesh clients located at: {:?}", mesh_clients);
}

fn main() {
    firefly_algorithm();
}
