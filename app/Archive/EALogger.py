import csv
import statistics
import matplotlib.pyplot as plt

class EALogger:
    """Manages the recording of statistics and the generation of graphs."""
    
    def __init__(self):
        # list to stock stats of each generation
        self.history = []

    def log_generation(self, gen, population):
        """Calculates and records the statistics (best, average, worst) for a given generation."""
        fitnesses = [ind.fitness for ind in population]
        
        best = min(fitnesses)
        worst = max(fitnesses)
        avg = statistics.mean(fitnesses)
        
        self.history.append({
            'generation': gen,
            'best': best,
            'avg': avg,
            'worst': worst
        })

    def export_to_csv(self, filename="ea_results.csv"):
        """Export the history to a CSV file."""
        if not self.history:
            print("No data.")
            return
            
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['generation', 'best', 'avg', 'worst'])
            writer.writeheader()
            writer.writerows(self.history)
        print(f"Resultats welle exported in {filename}")

    def plot_progress(self):
        """Generates and displays a graph showing changes in fitness."""
        if not self.history:
            print("No data.")
            return

        gens = [data['generation'] for data in self.history]
        bests = [data['best'] for data in self.history]
        avgs = [data['avg'] for data in self.history]
        worsts = [data['worst'] for data in self.history]

        plt.figure(figsize=(10, 6))
        plt.plot(gens, bests, label='Best', color='green', linewidth=2)
        plt.plot(gens, avgs, label='Average', color='blue', linestyle='--')
        plt.plot(gens, worsts, label='Worst', color='red', alpha=0.5)

        plt.title("The Evolution of the Objective Function Across Generations")
        plt.xlabel("Generations")
        plt.ylabel("Total flow time (Fitness)")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_save(self, save_path):
        if not self.history:
            print("No data.")
            return

        gens = [data['generation'] for data in self.history]
        bests = [data['best'] for data in self.history]
        avgs = [data['avg'] for data in self.history]
        worsts = [data['worst'] for data in self.history]

        plt.figure(figsize=(10, 6))
        plt.plot(gens, bests, label='Best', color='green', linewidth=2)
        plt.plot(gens, avgs, label='Average', color='blue', linestyle='--')
        plt.plot(gens, worsts, label='Worst', color='red', alpha=0.5)

        plt.title("The Evolution of the Objective Function Across Generations")
        plt.xlabel("Generations")
        plt.ylabel("Total flow time (Fitness)")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, bbox_inches='tight')
        
        plt.close()