# app/runners/__test_benchmark.py

import numpy as np


def test_one_max_contract():
    from app.utilities.benchmark import OneMaxProblem
    p = OneMaxProblem(50)
    assert p.name == "OneMax"
    assert p.num_features == 50
    assert p.dim == 50
    assert p.evaluate([1] * 50) == 0.0
    assert p.evaluate([0] * 50) == 50.0
    batch = p.evaluate_batch([[1]*50, [0]*50])
    assert batch[0] == 0.0
    assert batch[1] == 50.0
    p.reset_counter()
    assert p.evaluations_count == 0
    print("  PASS OneMax contract")


def test_existing_benchmarks():
    from app.utilities.benchmark import SphereProblem
    from app.differential_evolution.algorithm import DifferentialEvolution
    p = SphereProblem()
    config = {"popsize": 20, "max_generations": 10, "seed": 42, "CR": 0.7, "F1": 0.5}
    de = DifferentialEvolution(p, config)
    result = de.run()
    assert result.best_fitness < float("inf")
    assert p.num_features == 2
    assert p.evaluate([0, 0]) == 0.0
    print(f"  PASS Sphere benchmark: fitness={result.best_fitness:.6f}")


def test_sa_on_one_max():
    from app.utilities.benchmark import OneMaxProblem
    from app.simulated_annealing.algorithm import SimulatedAnnealing
    p = OneMaxProblem(50)
    sa = SimulatedAnnealing(p, max_evaluations=500, initial_temp=100.0, cooling_rate=0.99, seed=42)
    result = sa.run()
    assert result.best_fitness < 50.0
    assert result.best_fitness >= 0.0
    assert len(result.best_mask) == 50
    print(f"  PASS SA on OneMax(50): fitness={result.best_fitness:.4f}")


def test_nbpso_on_one_max():
    from app.utilities.benchmark import OneMaxProblem
    from app.particle_swarm.algorithm import NovelBinaryParticleSwarmOptimization
    p = OneMaxProblem(50)
    nbpso = NovelBinaryParticleSwarmOptimization(p, {"swarm_size": 20, "max_generations": 50, "seed": 42, "w": .8, "c1": 1.9, "c2": 1, "vmax": 4.0})
    result = nbpso.run()
    assert result.best_fitness < 50.0
    assert len(result.best_mask) == 50
    print(f"  PASS NBPSO on OneMax(50): fitness={result.best_fitness:.4f}")


def test_de_on_one_max():
    from app.utilities.benchmark import OneMaxProblem
    from app.decoders import build_decoder
    from app.differential_evolution.featureselection import DecodedFeatureSelectionProblem
    from app.differential_evolution.algorithm import DifferentialEvolution
    p = OneMaxProblem(50)
    decoder = build_decoder({"name": "sigma", "bounds": [-6.0, 6.0]})
    ds = DecodedFeatureSelectionProblem(p, decoder)
    config = {"popsize": 20, "max_generations": 30, "seed": 42, "strategy": "current-to-rand/1", "CR": 0.5, "F1": 0.4}
    de = DifferentialEvolution(ds, config)
    result = de.run()
    assert result.best_fitness < 50.0
    assert result.best_mask is not None
    print(f"  PASS DE+Sigma on OneMax(50): fitness={result.best_fitness:.4f}")

def test_nbpso_on_one_max_ldiw(config):
    from app.utilities.benchmark import OneMaxProblem
    from app.particle_swarm.algorithm import NovelBinaryParticleSwarmOptimization
    results = []
    avg = np.mean([r.best_fitness for r in results])
    assert result.best_fitness < 50.0
    assert len(result.best_mask) == 50
    print(f"  PASS NBPSO with LDIW on OneMax(50): fitness={result.best_fitness:.4f}")
    return result


if __name__ == "__main__":
    print("Running verification tests...")
    #test_one_max_contract()
    #test_existing_benchmarks()
    #test_sa_on_one_max()
    test_nbpso_on_one_max()
    configs =  [{"swarm_size": 20, "max_generations": 50, "seed": 42, "w": {"w_max": x, "w_min": y}, "c1": 1.9, "c2": 1, "vmax": 4.0} for x, y in np.dstack(np.meshgrid(np.linspace(0.6, 1.2, 10), np.linspace(0.4, 0.9, 10))).reshape(-1, 2)]
        
    results = list(test_nbpso_on_one_max_ldiw(config) for config in configs)
    for config, result in zip(configs, results):
        print(f"NBPSO for w_max={config['w']['w_max']:.2f}, w_min={config['w']['w_min']:.2f}: fitness={result.best_fitness:2.0f}")
    #test_de_on_one_max()
    print("ALL TESTS PASSED")
    