use rand::Rng;
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use serde_json::json;

const NUMBER_OF_MESH_ROUTERS: usize = 16;
const NUMBER_OF_MESH_CLIENTS: usize = 32;
const DIMENSIONS: usize = 2;
const NUMBER_OF_ITERATIONS: usize = 100;
const ALPHA: f64 = 0.5;
const BETA0: f64 = 1.0;
const GAMMA: f64 = 1.0;
const LOWER_BOUND: f64 = 0.0;
const UPPER_BOUND: f64 = 32.0;
const MAXIMUM_COMMUNICATION_DISTANCE: f64 = 4.5;

// Fitness Weights
const PRIORITY_SGC: f64 = 0.8;
const PRIORITY_NCMC: f64 = 0.1;
const PRIORITY_NCMCPR: f64 = 0.1;

// Distance function
fn distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| (xi - yi).powi(2)).sum::<f64>().sqrt()
}

// Function to compute Size of Giant Component (SGC)
fn sgc(routers: &[[f64; DIMENSIONS]]) -> usize {
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
                        if dist <= MAXIMUM_COMMUNICATION_DISTANCE {
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
    largest_component
}

// Function to compute Number of Covered Mesh Clients (NCMC)
fn ncmc(routers: &[[f64; DIMENSIONS]], clients: &[[f64; DIMENSIONS]]) -> usize {
    let mut covered_clients = 0;
    for client in clients {
        for router in routers {
            if distance(router, client) <= MAXIMUM_COMMUNICATION_DISTANCE {
                covered_clients += 1;
                break;
            }
        }
    }
    covered_clients
}

// Function to compute Number of Covered Mesh Clients per Router (NCMCpR)
fn ncmcpr(routers: &[[f64; DIMENSIONS]], clients: &[[f64; DIMENSIONS]]) -> f64 {
    ncmc(routers, clients) as f64 / routers.len() as f64
}

// Fitness function
fn fitness_function(routers: &[[f64; DIMENSIONS]], clients: &[[f64; DIMENSIONS]]) -> f64 {
    let sgc = sgc(routers) as f64;
    let ncmc = ncmc(routers, clients) as f64;
    let ncmcpr = ncmcpr(routers, clients);

    (PRIORITY_SGC * sgc) + (PRIORITY_NCMC * ncmc) + (PRIORITY_NCMCPR * ncmcpr)
}

// Save results to file
fn save_results(
    routers: &Vec<[f64; DIMENSIONS]>,
    clients: &Vec<[f64; DIMENSIONS]>,
    best_fitness: f64,
    sgc: usize,
    ncmc: usize,
    ncmcpr: f64,
) {
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

// Firefly Algorithm
fn firefly_algorithm() {
    let mut rng = rand::thread_rng();
    let mut mesh_routers = vec![[0.0; DIMENSIONS]; NUMBER_OF_MESH_ROUTERS];
    let mut mesh_clients = vec![[0.0; DIMENSIONS]; NUMBER_OF_MESH_CLIENTS];

    // Initialize mesh clients randomly
    for client in mesh_clients.iter_mut() {
        for coord in client.iter_mut() {
            *coord = rng.gen_range(LOWER_BOUND..UPPER_BOUND);
        }
    }

    // Initialize mesh routers randomly
    for router in mesh_routers.iter_mut() {
        for coord in router.iter_mut() {
            *coord = rng.gen_range(LOWER_BOUND..UPPER_BOUND);
        }
    }

    let mut best_mesh_routers = mesh_routers.clone();
    let mut best_fitness = fitness_function(&mesh_routers, &mesh_clients);

    // Firefly Algorithm Iterations
    for _ in 0..NUMBER_OF_ITERATIONS {
        for i in 0..NUMBER_OF_MESH_ROUTERS {
            for j in 0..NUMBER_OF_MESH_ROUTERS {
                if i != j {
                    let r_ij = distance(&mesh_routers[i], &mesh_routers[j]);
                    let beta = BETA0 * (-GAMMA * r_ij * r_ij).exp();

                    for d in 0..DIMENSIONS {
                        let attraction = beta * (mesh_routers[j][d] - mesh_routers[i][d]);
                        let randomness = ALPHA * (rng.r#gen::<f64>() - 0.5);

                        mesh_routers[i][d] += attraction + randomness;
                        mesh_routers[i][d] = mesh_routers[i][d].clamp(LOWER_BOUND, UPPER_BOUND);
                    }
                }
            }
        }

        let current_fitness = fitness_function(&mesh_routers, &mesh_clients);
        if current_fitness > best_fitness {
            best_fitness = current_fitness;
            best_mesh_routers = mesh_routers.clone();
        }
    }

    // Save and print results
    let sgc_value = sgc(&best_mesh_routers);
    let ncmc_value = ncmc(&best_mesh_routers, &mesh_clients);
    let ncmcpr_value = ncmcpr(&best_mesh_routers, &mesh_clients);
    save_results(&best_mesh_routers, &mesh_clients, best_fitness, sgc_value, ncmc_value, ncmcpr_value);

    println!("Final Fitness Score: {}", best_fitness);
    println!("Results saved to firefly_results.json");
}

// Main Function
fn main() {
    firefly_algorithm();
}
