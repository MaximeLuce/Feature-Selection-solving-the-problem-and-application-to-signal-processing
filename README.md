# Feature-Selection-solving-the-problem-and-application-to-signal-processing
Feature Selection : solving the problem and application to signal processing

## Context of this project

This project is a part of the course "Optimization Methods: Theory and Applications" at the Wroclaw Univeristy of Science and Technology (project).

It aims to study the Feature Selection which is an NP-hard problem. The purpose of this project is to compare efficently different algorithms to solve it. We will compare:

- Simulated Annealing (SA) with two parameters
  - initial temperature
  - cooling rate
- Differential Evolution (DE) (to be continued)
- Particle Swarm Optimization (PSO) (to be continued)

(MAYBE)
- random search (maybe)
- Evolutionary Algorithm (EA) with different parameters (maybe)
  - Population size
  - Crossover probability
  - Mutation probability
  - Tour size
  - Number of generation
  - Crossover methods : OX and PMX
  - Mutation methods : Swap or Inversion

In order to do that, we will use (and compare ?) Machine Learning:

- kNN for light model
- SVM for heavy model used as a baseline (comparison with others ?)

## Use

1) Load the project
2) On the main folder, run `python -m app.main`
3) If you want to access a specific file (e.g., a Test File), use `python -m app.Tests.TestSA`

## Datasets

Description TO DO

1) https://archive.ics.uci.edu/dataset/174/parkinsons
2) https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
3) https://archive.ics.uci.edu/dataset/52/ionosphere
4) https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
5) https://www.kaggle.com/datasets/meruvakodandasuraj/sonar-dataset-underwater-object-classification?resource=download

More infos on UCI https://github.com/uci-ml-repo/ucimlrepo

## Naviguation

- `app`: all the pythons file of the project
    - `Data`: instance of the problem (commin from [GitHub Pages]([https://pages.github.com/](https://github.com/chneau/go-taillard/tree/master/pfsp/instances))
    - `Utilities`: EALogger, ExperimentRunner
    - `Tests`: Unit tests on the class
    - `Problem`: DataLoader, Individual, Problem
    - `OptimizationAlgorithm`: EvolutionaryAlgorithm, OptimizationAlgorithm, RandomSearch
    - `Results`: folder where all csv were exported and contains file to export CSV to LaTeX tables
    - `Archive`: old files
- `Topic`: PDF files of the exercise goal and description
- `Report`: Report on the project
- `Figures`: Convergence graphs and figures
- `Notes`: some of my notes during the project

### Modules used for this project

#### Standard and external modules

#### Built-ins

#### Methods for lists, strings and files
