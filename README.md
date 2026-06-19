# Feature Selection

Comparative study of optimization algorithms for the **Feature Selection** problem. Part of the *Optimization Methods: Theory and Applications* course at Wrocław University of Science and Technology.

## Algorithms

| Algorithm | Type | Key Parameters |
|-----------|------|----------------|
| **BDE** — Binary Differential Evolution | Population (DE) | $strategy, NP, G, F, CR$|
| **AMDE** — Angle Modulation DE | Population (DE) | $strategy, NP, G, F, C |
| **SA** — Simulated Annealing | Single-solution | $T_0$, $\alpha$ |
| **NBPSO** — Novel Binary PSO | Swarm | S, G,  |
| **NBPSO-LDIW** — NBPSO with linearly decreasing inertia | Swarm | $S, G, \lambda, c_1, c_2$ 

## Evaluators

- **SVM** (heavy) — `StandardScaler` + `SVC(RBF)`, $C=1.0, \gamma=scale, 5-fold CV$
- **k-NN** (light) — `KNeighborsClassifier`, $k=5$, 5-fold $CV$

## Datasets

| ID | Name | Features | Instances | Source |
|----|------|----------|-----------|--------|
| 174 | Parkinsons | 22 | 197 | [UCI](https://archive.ics.uci.edu/dataset/174/parkinsons) |
| 17 | Breast Cancer Wisconsin | 30 | 569 | [UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| 52 | Ionosphere | 34 | 351 | [UCI](https://archive.ics.uci.edu/dataset/52/ionosphere) |
| 151 | Sonar: Mines vs Rocks | 60 | 208 | [UCI](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) |

## Project Structure

```
app/
  core/                    # config, I/O helpers, monitoring
  problem/                 # dataset loading, fitness, evaluators, masks
  decoders/                # binary decoders (sigma, angle modulation)
  differential_evolution/  # DE algorithms
  simulated_annealing/     # SA algorithm
  particle_swarm/          # NBPSO algorithm
  runners/                 # parameter sweeps, analysis, plots
  Results/                 # raw JSONL run data, summary CSVs
  Data/                    # dataset CSV files (ultimately not used)
  config.json              # experiment configurations for runs
```

## Usage

```shell
# run experiments defined in config.json (modify output paths and config groups!)
python -m app.runners.de_parameters

python -m app.runners.sa_parameters

python -m app.runners.nbpso_parameters

# run analysis defined in main
python -m app.runners.analysis

```
## Results

Tables used to generate tables in the report are available in `app/Results/summary/`.

Data for analysis in `app/Results/raw/` is too voluminous to be included.

Best found solutions are saved in `best_solutions.csv`.

Report placed in `report.pdf`.
