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
