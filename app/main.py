## security of number of core
import os
#os.environ["OMP_NUM_THREADS"] = "1" # OpenMP
#os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS
#os.environ["MKL_NUM_THREADS"] = "1" # MKL
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # Accelerate
#os.environ["NUMEXPR_NUM_THREADS"] = "1" # NumExpr


from app.runners.best_runner_sa_ea import BestRunnerSAEA
from app.runners.comparison_runner import ComparisonRunner
from app.runners.ea_convergence import EAConvergence
from app.runners.ea_parameters import EAParameters
from app.runners.sa_parameters import SAParameters
from app.runners.de_parameters import DEParameters, DEParametersParallell

if __name__ == "__main__":
    #runner = EAParameters()
    #runner = SAParameters()
    #runner = ComparisonRunner()
    #runner = BestRunnerSAEA()
    #runner = EAConvergence()
    #runner = DEParameters()
    runner = DEParametersParallell()
    runner.run_all()
