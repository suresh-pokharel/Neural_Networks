
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

    # the point at which crossover is made between two parents ie. take mid point for now
    crossover_point = np.uint8(size_of_offsprings[1]/2)

    for k in range(size_of_offsprings[0]):
        # index of first parent
        parent_index_1 = k % parents.shape[0]
        parent_index_2 = (k + 1) % parents.shape[0]

        # now the new offspring will have 1st half from parent_1 and second half from parent_2
        # for kth offspring, copy values from  parent1, from index 0 to crossover_point
        offspring[k,0:crossover_point] = parents[parent_index_1,0:crossover_point]

        # for 2nd half of kth offspring, copy values from  parent2, from index crossover_point to size_of_offsprings[1]
        offspring[k,crossover_point:] = parents[parent_index_2,crossover_point:]

    return offspring


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


def mutation(offsprings):
    for k in range(offsprings.shape[0]):
        #  add some random value (say, between -1 and 1) at random position for each offspring
        random_value = np.random.uniform(-1,1,1)

        # We have a offspring like [-3.15639367  0.00461411 - 3.25848937  2.4363469   2.17465714 - 1.16898281]
        # We generate random index between 0 and 5
        random_index = np.random.randint(0, offsprings.shape[1], 1)

        # add random_value to that element
        offsprings[k][random_index] += random_value

        # Offsprings after crossover
        # [[-0.30508167  1.42462414 - 0.81328178 - 2.96422194 - 0.44369271 - 3.9921851]
        #  [2.01979429  2.60355749 - 3.47669277 - 1.37758176 - 0.26996857  1.6751438]
        # [0.78666558 - 3.64939753 - 0.8449585 - 2.39223334 1.45788797 - 1.34218045]
        # [2.2665711   3.36623223  2.15307296  3.72525495 - 0.25271673  1.21482798]]
        # random_value[0.82415308]
        # random_index[2]
        # random_value[0.4321758]
        # random_index[1]
        # random_value[0.49525073]
        # random_index[0]
        # random_value[0.56047718]
        # random_index[2]
        # Offsprings after mutation
        # [[-0.30508167  1.42462414  0.0108713 - 2.96422194 - 0.44369271 - 3.9921851]
        #  [2.01979429  3.03573329 - 3.47669277 - 1.37758176 - 0.26996857  1.6751438]
        # [1.28191632 - 3.64939753 - 0.8449585 - 2.39223334  1.45788797 - 1.34218045]
        # [2.2665711   3.36623223  2.71355014  3.72525495 - 0.25271673  1.21482798]]

    return offsprings


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
best_outputs = []

# number of iterations/generations to be computed
no_generations = 500

# target fitness: expected fitness after generations
target_fitness = 100

# number of parents mating
no_parents_mating = 4

# Apply the GA variants (crossover and mutation) to produce the offspring
# of the next generation, creating the new population by appending both
# parents and offspring, and repeating such steps for a number of iterations
for generation in range(no_generations):
    # calculate fitness for
    print("_____________________________________________________________________")
    print("Generation: ", generation)

    # calculate fitness of each chromosome
    # sample fitness
    # [-23.97840031  22.80622622 - 34.23968544  36.8922263 - 5.09026912
    #  - 16.40461743 - 1.01066003 - 15.91178617]
    fitness = calculate_fitness(initial_inputs, new_population)
    print("Fitness:\n", fitness)
    print("Best fitness at this generation: ",np.max(fitness))

    # stop if the required fitness is obtained. say 100 is what we need
    if np.max(fitness) >= target_fitness:
        print("..........................................................................")
        print("   Target fitness obtained at generation", generation)
        print("..........................................................................")
        break

    # select the best parents for next generation
    # [[-1.70911178  3.41416114  0.76248373  1.3703949 - 3.6846485   2.03017764]
    # [3.09929308 - 1.91522523  1.16152566 - 2.96669409 - 0.47461211 - 1.35182463]
    # [-0.84403851  # 1.12669114  # 1.3089964   # 1.67528352 - 0.333356 - 0.06183099]
    # [-0.64943855  2.88827367 - 2.20031029  3.61630038 - 0.06633548 - 1.6242148]]
    selected_parents = select_best_parents(new_population, fitness, no_parents_mating)

    # generate next generation by applying crossover operation over fittest offsprings
    # Parents: 0
    # [[-0.2867649   0.93408083 - 1.65540785  2.91761097 - 0.85340808 - 2.4815444]
    #  [-3.15639367  0.00461411 - 3.25848937 - 3.13084983 - 3.52819968 - 0.60952906]
    # [1.93088492  1.25030287  0.47903939  2.4363469  2.17465714 - 1.16898281]
    # [-1.89157854  0.66699253  2.93577152  3.24048577  1.43398486  0.47573811]]

    # Offsprings after crossover
    # [[-0.2867649   0.93408083 - 1.65540785 - 3.13084983 - 3.52819968 - 0.60952906]
    #  [-3.15639367  0.00461411 - 3.25848937  2.4363469   2.17465714 - 1.16898281]
    #  [1.93088492 1.25030287  0.47903939  3.24048577  1.43398486  0.47573811]
    # [-1.89157854  0.66699253  2.93577152  2.91761097 - 0.85340808 - 2.4815444]]
    offsprings_after_crossover = crossover(selected_parents, (population_size[0] - selected_parents.shape[0], num_weights))

    # mutate the offspring produced after crossover
    offsprings_after_mutation = mutation(offsprings_after_crossover)

    # make the new population using selected parents and newly generated offsprings
    new_population[0:selected_parents.shape[0]] = selected_parents
    new_population[selected_parents.shape[0]:] = offsprings_after_mutation

    # calculate fitness after generations
    fitness = calculate_fitness(initial_inputs, new_population)

# display result in matplotlib


# Taking our expected fitness as 100, Here is a output, where we can see the increase in best value at each iteration.
# In worst case, the best value remains same but never decreases.


#         _____________________________________________________________________
#         Generation:  0
#         Fitness:
#          [ 13.93984769 -25.65878239  57.17891522 -39.43519062  14.90653709
#           50.86832354 -28.24467152  41.37612263]
#         Best fitness at this generation:  57.17891521880241
#         _____________________________________________________________________
#         Generation:  1
#         Fitness:
#          [57.17891522 50.86832354 41.37612263 14.90653709 69.41004558 20.24179958
#          43.09122529 22.47932244]
#         Best fitness at this generation:  69.4100455752385
#         _____________________________________________________________________
#         Generation:  2
#         Fitness:
#          [69.41004558 57.17891522 50.86832354 43.09122529 57.17480957 71.8929024
#          29.21798609 63.21398367]
#         Best fitness at this generation:  71.89290239886336
#         _____________________________________________________________________
#         Generation:  3
#         Fitness:
#          [71.8929024  69.41004558 63.21398367 57.17891522 65.6806793  72.92745848
#          49.40366661 75.33316767]
#         Best fitness at this generation:  75.33316767036955
#         _____________________________________________________________________
#         Generation:  4
#         Fitness:
#          [75.33316767 72.92745848 71.8929024  69.41004558 72.02337457 76.29539503
#          66.05210365 85.83605501]
#         Best fitness at this generation:  85.8360550136698
#         _____________________________________________________________________
#         Generation:  5
#         Fitness:
#          [85.83605501 76.29539503 75.33316767 72.92745848 68.12027428 79.57425551
#          71.6080107  90.19448914]
#         Best fitness at this generation:  90.19448914013537
#         _____________________________________________________________________
#         Generation:  6
#         Fitness:
#          [90.19448914 85.83605501 79.57425551 76.29539503 91.89879803 74.00922648
#          71.78164625 89.3264923 ]
#         Best fitness at this generation:  91.8987980298065
#         _____________________________________________________________________
#         Generation:  7
#         Fitness:
#          [91.89879803 90.19448914 89.3264923  85.83605501 95.67073164 90.46713604
#          90.99158746 82.74475492]
#         Best fitness at this generation:  95.67073163592269
#         _____________________________________________________________________
#         Generation:  8
#         Fitness:
#          [95.67073164 91.89879803 90.99158746 90.46713604 92.6846689  95.39572162
#          99.13898177 91.71124964]
#         Best fitness at this generation:  99.13898176580247
#         _____________________________________________________________________
#         Generation:  9
#         Fitness:
#          [99.13898177 95.67073164 95.39572162 92.6846689  93.70791724 93.09411907
#          88.7999642  91.27934243]
#         Best fitness at this generation:  99.13898176580247
#         _____________________________________________________________________
#         Generation:  10
#         Fitness:
#          [ 99.13898177  95.67073164  95.39572162  93.70791724  95.89974598
#           95.24585079  97.30937667 104.48693703]
#         Best fitness at this generation:  104.48693703168905
#         ..........................................................................
#          Target fitness obtained at generation 10
#         ...........................................................................
