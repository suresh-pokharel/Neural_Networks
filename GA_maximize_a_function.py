

# GA to maximize a function
import numpy as np


# class GACustom:
# def __init__(self, ):
#     self.inputs = inputs
#     self.weights = weights

def calculate_fitness(inputs, weights):
    # Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6 + ...
    # axis = 1 adds column wise
    return np.sum(inputs * weights, axis=1)


def crossover(parents, size_of_offsprings):
    # finds the offspring from the selected parents
    # size_of_offspring is number of offspring to produce from selected parents

    # create empty array of size size_of_offsprings
    offspring = np.zeros(size_of_offsprings)

    # the point at which crossover is made between two parents
    crossover_point = np.uint8()





def select_best_parents(population, fitness_values, no_of_parents):
    # create a empty matrix of size no_of_parents * sol_per_population ie. 4*8
    parents = np.zeros((no_of_parents, population.shape[1]))

    for parent in range(no_of_parents):
        # find the index of array at which maximum fitness is in format array([4],)
        max_fitness_index = np.where(fitness_values == np.max(fitness_values))

        # Extract only integer from array([4],) ie. returns 4
        max_fitness_index = max_fitness_index[0][0]

        # copy the population with maximum fitness to parents
        parents[parent, :] = population[max_fitness_index, :]

        # assign very small number to max index so that next largest value becomes max in next iteration
        fitness_values[max_fitness_index] = -99999999.99

    return parents


# initial inputs
initial_inputs = [4, -2, 3.5, 5, -11, -4.7]

# number of weights
num_weights = 6

# solution per population
sol_per_population = 8

# population_size
# each population has sol_per_population chromosome and each chromosome has num_weights genes
population_size = (sol_per_population, num_weights)

# initialize the populations with random values
# sample new_population 8*6 size
# [[-0.1431349   2.98822158  1.75406617  2.00163161 -1.9038607  -1.36200501]
# #  [ 3.069945    2.60337114 -3.55413413 -1.8373041  -2.85284786 -3.66674658]
# #  [-2.12333595  0.08144516 -2.87172463 -0.71608823  2.91292308 -3.29653688]
# #  [ 3.65391504  1.5657035  -3.29691704  0.62768474  1.0882381   1.84810325]
# #  [-0.90651377 -3.94351897  0.08131775  2.09383602 -0.5152794  -1.13899767]
# #  [ 2.01303422 -1.24593983 -1.06046591  2.78473386 -3.73402672 -1.9699166 ]
# #  [ 1.64744046  1.39978208  1.6619033   1.92679263 -1.65791437  2.66836236]
# #  [ 0.28210299 -2.40772116  1.15015544  1.17031268 -2.83846519 -2.20636383]]
new_population = np.random.uniform(-4.0, 4.0, population_size)
print(new_population)
best_outputs = []

# number of iterations/generations to be computed
no_generations = 2

# number of parents mating
no_parents_mating = 4

# Apply the GA variants (crossover and mutation) to produce the offspring
# of the next generation, creating the new population by appending both
# parents and offspring, and repeating such steps for a number of iterations
for generation in range(no_generations):
    # calculate fitness for
    print("Generation: ", generation)

    # calculate fitness of each chromosome
    # sample fitness
    # [-23.97840031  22.80622622 - 34.23968544  36.8922263 - 5.09026912
    #  - 16.40461743 - 1.01066003 - 15.91178617]
    fitness = calculate_fitness(initial_inputs, new_population)

    # select the best offspring as a parent for next generation
    # [[-1.70911178  3.41416114  0.76248373  1.3703949 - 3.6846485   2.03017764]
    # [3.09929308 - 1.91522523  1.16152566 - 2.96669409 - 0.47461211 - 1.35182463]
    # [-0.84403851  # 1.12669114  # 1.3089964   # 1.67528352 - 0.333356 - 0.06183099]
    # [-0.64943855  2.88827367 - 2.20031029  3.61630038 - 0.06633548 - 1.6242148]]
    selected_parents = select_best_parents(new_population, fitness, no_parents_mating)



    print(selected_parents)

    # generate next generation by applying crossover operation over fittest offsprings
    # offsprings = crossover(selected_parents,)
