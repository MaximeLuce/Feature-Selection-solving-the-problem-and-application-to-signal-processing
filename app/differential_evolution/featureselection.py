# featureselection.py

class FeatureSelectionProblem:
    def __init__(self, problem, decoder):
        self.problem = problem
        self.decoder = decoder
        self.bounds = decoder.bounds(problem.num_features)

    @property
    def dim(self):
        return len(self.bounds)

    @property
    def evaluations_count(self):
        return self.problem.evaluations_count

    def reset(self):
        self.problem.reset_counter()

    def decode(self, vector):
        return self.decoder.decode(vector, self.problem.num_features)

    def evaluate(self, vector):
        return self.problem.evaluate(self.decode(vector))
