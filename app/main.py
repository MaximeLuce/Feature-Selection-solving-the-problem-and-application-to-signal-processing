## security of number of core
#os.environ["OMP_NUM_THREADS"] = "1" # OpenMP
#os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS
#os.environ["MKL_NUM_THREADS"] = "1" # MKL
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # Accelerate
#os.environ["NUMEXPR_NUM_THREADS"] = "1" # NumExpr


from app.Archive.best_runner_sa_ea import BestRunnerSAEA
from app.Archive.comparison_runner import ComparisonRunner
from app.Archive.ea_convergence import EAConvergence
from app.Archive.ea_parameters import EAParameters
from app.runners.sa_parameters import SAParameters
from app.runners.de_parameters import DEParameters
from app.runners.nbpso_parameters import NBPSOParameters

if __name__ == "__main__":
    #runner = EAParameters()
    #runner = SAParameters()
    #runner = ComparisonRunner()
    #runner = BestRunnerSAEA()
    #runner = EAConvergence()
    #runner = DEParameters()
    #runner = NBPSOParameters()
    runner = DEParameters()
    runner.run_all()
    