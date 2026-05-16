import numpy as np
from app.OptimizationAlgorithm.OptimizationAlgorithm import OptimizationAlgorithm
from app.Problem.Individual import Individual

class Greedy(OptimizationAlgorithm):
    def __init__(self, problem):
        super().__init__(problem)

    def _get_completion_time(self, partial_sequence):
        """
        Compute c(x_j, M) for the last task in the sub-sequence. (completion time on the last machine)
        """
        if not partial_sequence:
            return 0
        
        M = self.problem.num_machines
        J = len(partial_sequence) # We use the length of the partial sequence
        p = self.problem.processing_times
        
        # completion time matrix
        C = np.zeros((M, J), dtype=int)
        
        # First task, first machine
        C[0][0] = p[0][partial_sequence[0]]
        
        # Other tasks, first machine
        for j in range(1, J):
            C[0][j] = C[0][j-1] + p[0][partial_sequence[j]]
            
        # first task on other machines
        for m in range(1, M):
            C[m][0] = C[m-1][0] + p[m][partial_sequence[0]]
            
        # rest of the tasks on the rest of the machines
        for m in range(1, M):
            for j in range(1, J):
                C[m][j] = max(C[m][j-1], C[m-1][j]) + p[m][partial_sequence[j]]
                
        # final time of the last task on the last machine
        return C[M-1][J-1]

    def run(self):
        """Executes the greedy method in a deterministic manner."""
        
        # list of unscheduled jobs
        unscheduled_jobs = list(range(self.problem.num_jobs))
        final_sequence = []
        
        while unscheduled_jobs:
            best_job = None
            best_completion_time = float('inf')
            
            # we try each unscheduled jobs
            for job in unscheduled_jobs:
                test_sequence = final_sequence + [job]
                
                # partial evaluation of the remaining job
                completion_time = self._get_completion_time(test_sequence)
                
                # memorize the best local solution
                if completion_time < best_completion_time:
                    best_completion_time = completion_time
                    best_job = job
                    
            # finally, adding the best solution to the sequence
            final_sequence.append(best_job)
            # and removing from unscheduled_jobs
            unscheduled_jobs.remove(best_job)
            
        # then we create the associate Individual
        greedy_individual = Individual(final_sequence)
        
        # we evaluate finally the individual
        greedy_individual.evaluate(self.problem)
        
        return greedy_individual