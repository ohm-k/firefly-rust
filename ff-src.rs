use rand::Rng;
use std::f64::consts::PI;

const N: usize = 20; // Number of fireflies
const D: usize = 2;  // Dimension of the problem
const number_of_simulations: usize = 100; // Maximum number of iterations
const ALPHA: f64 = 0.5; // Randomness parameter
const BETA0: f64 = 1.0; // Attractiveness constant
const GAMMA: f64 = 1.0; // Light absorption coefficient
const LOWER_BOUND: f64 = -10.0;
const UPPER_BOUND: f64 = 10.0;

// Objective function: Sphere function
fn objective_function(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

// Distance between two fireflies - Cartesian Distance
fn distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| (xi - yi).powi(2)).sum::<f64>().sqrt()
}

fn firefly_algorithm() {
    let mut rng = rand::thread_rng();
    let mut fireflies = vec![[0.0; D]; N];
    let mut brightness = vec![0.0; N];

    // Initialize fireflies randomly
    for i in 0..N {
        for j in 0..D {
            fireflies[i][j] = rng.gen_range(LOWER_BOUND..UPPER_BOUND);
        }
        brightness[i] = objective_function(&fireflies[i]);
    }

    for _ in 0..number_of_simulations {
        // Sort fireflies based on brightness (lower is better)
        let mut indices: Vec<usize> = (0..N).collect();
        indices.sort_by(|&i, &j| brightness[i].partial_cmp(&brightness[j]).unwrap());
        fireflies = indices.iter().map(|&i| fireflies[i]).collect();
        brightness = indices.iter().map(|&i| brightness[i]).collect();

        for i in 0..N {
            for j in 0..N {
                if brightness[j] < brightness[i] {
                    let r = distance(&fireflies[i], &fireflies[j]);
                    let beta = BETA0 * (-GAMMA * r * r).exp();
                    for d in 0..D {
                        fireflies[i][d] += beta * (fireflies[j][d] - fireflies[i][d]) +
                            ALPHA * (rng.gen::<f64>() - 0.5);
                        fireflies[i][d] = fireflies[i][d].clamp(LOWER_BOUND, UPPER_BOUND);
                    }
                    brightness[i] = objective_function(&fireflies[i]);
                }
            }
        }
    }

    // Print best solution found
    let best_index = brightness.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    println!("Best solution: {:?}", fireflies[best_index]);
    println!("Best objective value: {}", brightness[best_index]);
}

fn main() {
    firefly_algorithm();
}
