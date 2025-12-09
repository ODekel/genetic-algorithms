"""
Run the genetic algorithm for RNA-seq deconvolution.

This function performs cell type deconvolution on RNA-seq data using a genetic algorithm.
For each sample (column) in the gene expression matrix, it evolves a population of
cell type proportion vectors to maximize fitness (Pearson correlation with observed
expression, normalized by constraint satisfaction).

The function generates two visualization plots:
1. Shows average fitness per generation for each sample, with shaded regions indicating min-max range.
2. Shows overall average fitness across all samples per generation, with shaded region for min-max range.

The best solution for each sample is combined into a result matrix and saved to 'result.tsv'.

Outputs:
    - results.tsv: Tab-separated file with final cell type proportions

Console Output:
    - Evolution progress information
    - Mean final fitness of the result matrix
    - MSE (Mean Squared Error) between predicted and observed gene expression
    - Pearson correlation between predicted and observed gene expression
    - Result matrix statistics (mean and std of row sums)
"""
import json
import time
from collections.abc import Callable
from threading import Thread

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from genetic_algorithm import evolve
import crossover
import mutation
import selection


with open('./config.json', 'r') as config_file:
    config = json.load(config_file)


# Load data
m = pd.read_csv(config['sample_data_file'], sep='\t', header=0, index_col=0)
h = pd.read_csv(config['celltype_data_file'], sep='\t', header=0, index_col=0)
h = h.reindex(m.index)

mn = m.to_numpy(dtype=np.float32)
hn = h.to_numpy(dtype=np.float32)

# Storage for results and statistics
num_samples = mn.shape[1]
num_celltypes = hn.shape[1]
result = np.zeros((num_samples, num_celltypes), dtype=np.float32)

# Storage for fitness statistics per generation
# [sample_idx][generation] = (avg_fitness, max_fitness, min_fitness)
fitness_stats_per_sample = [[] for _ in range(num_samples)]


def _mean_squared_error_fitness(sample: npt.NDArray[np.float32], population: npt.NDArray[np.float32]
                                ) -> npt.NDArray[np.float32]:
    pred = hn @ population.T
    return (1.0 / ((sample - pred) ** 2).mean(axis=0)) / (np.abs(1.0 - population.sum(axis=1)) + 1.0)


def _set_result(sample_result: npt.NDArray[np.float32], out: npt.NDArray[np.float32], sample_row: int) -> None:
    out[sample_row] = sample_result


# Function to run evolution for a single sample
def run_sample(sample: npt.NDArray[np.float32], rng: np.random.Generator, cfg: dict,
               append_stats: Callable[[tuple[np.float32, np.float32, np.float32]], None],
               set_result: Callable[[npt.NDArray[np.float32]], None]):
    # Stop condition
    def stop_condition(pop: npt.NDArray[np.float32], fits: npt.NDArray[np.float32], generation: int):
        return generation >= cfg['max_generations']

    # Run evolution
    best_individual = None
    best_fitness = -np.inf

    for population, fitnesses in evolve(
        population=rng.random((cfg['population_size'], num_celltypes), dtype=np.float32),
        fitness=lambda p: _mean_squared_error_fitness(sample, p),
        parent_selector=lambda p, f, n: selection.roulette_wheel(p, f, n, replace=True, rng=rng),
        offsprings_per_generation=cfg['offsprings_per_generation'],
        parents_per_offspring=2,
        offspring_selector=lambda p, f, n: selection.roulette_wheel(p, f, n, replace=False, rng=rng),
        reproduce=lambda p, s, pc: crossover.n_point(p, s, pc, rng),
        first_parent_offset=lambda n: None,
        pc=cfg['pc'],
        mutate=lambda p, pm, vg: mutation.redetermination(p, pm, rng, vg),
        mutation_generator=lambda n: rng.random(n, dtype=np.float32),
        pm=cfg['pm'],
        stop=stop_condition
    ):
        # Track statistics
        avg_fit = fitnesses.mean()
        max_fit = fitnesses.max()
        min_fit = fitnesses.min()
        append_stats((avg_fit, max_fit, min_fit))

        # Track best individual
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_individual = population[fitnesses.argmax()].copy()

    # Store the best individual
    set_result(best_individual)


# Run evolution for each sample in parallel
print(f"Running genetic algorithm for {num_samples} samples...")
print(f"Parameters: max_generations={config['max_generations']}, population_size={config['population_size']}, "
      f"pc={config['pc']}, pm={config['pm']}, offsprings_per_generation={config['offsprings_per_generation']}")

start_time = time.time()
threads = []
for i in range(num_samples):
    rng = np.random.default_rng()
    thread = Thread(target=run_sample, args=(mn[:, i][:, None], rng, config,
                                             lambda s, i=i: fitness_stats_per_sample[i].append(s),
                                             lambda out, i=i: _set_result(out, result, i)))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
end_time = time.time()

print("Evolution complete!")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Create visualizations
print("Creating visualizations...")

# Graph 1: Per-sample fitness over generations
fig1, ax1 = plt.subplots(figsize=(12, 8))
generations = np.arange(config['max_generations'])

for sample_idx in range(num_samples):
    stats = np.array(fitness_stats_per_sample[sample_idx])
    avg_fitness = stats[:, 0]
    max_fitness = stats[:, 1]
    min_fitness = stats[:, 2]

    # Plot line for average fitness
    line = ax1.plot(generations, avg_fitness, label=f'Sample {sample_idx}', alpha=0.7)
    color = line[0].get_color()

    # Add shadow between max and min
    ax1.fill_between(generations, min_fitness, max_fitness, alpha=0.2, color=color)

ax1.set_xlabel('Generation')
ax1.set_ylabel('Fitness')
ax1.set_title('Average Fitness per Sample Over Generations')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# Graph 2: Overall fitness across all samples
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Calculate overall statistics across all samples
overall_avg = np.zeros(config['max_generations'])
overall_max = np.zeros(config['max_generations'])
overall_min = np.zeros(config['max_generations'])

for gen in range(config['max_generations']):
    gen_fitnesses = []
    for sample_idx in range(num_samples):
        stats = fitness_stats_per_sample[sample_idx][gen]
        # We use the average fitness of each sample for the overall calculation
        gen_fitnesses.append(stats[0])

    overall_avg[gen] = np.mean(gen_fitnesses)
    overall_max[gen] = np.max(gen_fitnesses)
    overall_min[gen] = np.min(gen_fitnesses)

# Plot line for average fitness
ax2.plot(generations, overall_avg, label='Average Fitness (across samples)',
         color='blue', linewidth=2)

# Add shadow between max and min
ax2.fill_between(generations, overall_min, overall_max, alpha=0.3, color='blue',
                 label='Min-Max Range')

ax2.set_xlabel('Generation')
ax2.set_ylabel('Fitness')
ax2.set_title('Overall Average Fitness Across All Samples Over Generations')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# Save result matrix to TSV
result_df = pd.DataFrame(
    result * 100.0,
    index=m.columns,
    columns=h.columns
)
result_df.to_csv(config['results_file'], sep='\t')
print(f"\nSaved: {config['results_file']}")

# Calculate final metrics
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

# Calculate mean fitness of result matrix
final_fitnesses = []
for i in range(num_samples):
    final_fitnesses.append(_mean_squared_error_fitness(mn, result))

mean_final_fitness = np.mean(final_fitnesses)
print(f"Mean Final Fitness: {mean_final_fitness:.6f}")

# Calculate MSE with mn
m_pred = hn @ result.T
mse = ((mn - m_pred) ** 2).mean()
print(f"MSE with original data: {mse:.6f}")

# Calculate Pearson correlation with mn
pearson = sp.stats.pearsonr(mn, m_pred, axis=None).statistic
print(f"Pearson Correlation with original data: {pearson:.6f}")

# Additional statistics
print(f"\nResult Matrix Statistics:")
print(f"Mean row sum: {result.sum(axis=1).mean():.6f}")
print(f"Std row sum: {result.sum(axis=1).std():.6f}")

print("="*60)
