import random
import numpy as np
from deap import base, creator, tools

# # Configuration

# +
# Importance given to the magnitude of the mutation in the fitness
MUT_SIZE_WEIGHT = 1
MUT_NUM_WEIGHT = 1

##### GENERATION PARAMETERS #####
# Initial population mutation params
INI_MEAN = 0
INI_STDEV = 0.01

##### MUTATION PARAMETERS #####
# Probability for mutating a individual
MUTPB = 0.2
# Probability for mutating each gene
INDPB = 0.1
# Optimization mutation params
MUT_MEAN = 0
MUT_STDEV = 0.1

##### REVERSE MUTATION PARAMETERS #####
# Probability for mutating a individual
MUTREVPB = 0.3
# Probability for mutating each gene
INDREVPB = 0.4

##### MATING PARAMETERS #####
# Probability with which two individuals are crossed
CXPB = 0.5

# Number of optimization iterations
N_GENS = 250


# -

# # Fitness and initial population functions

def evaluate_ind(ind, base_ind, model, label_obj, lab_list):
    # Objective 1: Classify a certain label
    ind = np.copy(np.array(ind))
    ind_mut = ind + base_ind
    pred = model.predict(ind_mut.reshape(1,500))
    confidence = pred[0][lab_list.index(label_obj)]
    
    # Objective 2: Reduce the magnitude of total mutations
    mut_size = np.sum(np.abs(ind))
    
    # Objective 3: Reduce the number of genes mutated
    mut_num = len(np.unique(ind)) - 1
    
    return confidence, mut_size, mut_num


def evaluate_pop(pop, base_ind, model, label_obj, lab_list):
    fits = []
    
    for ind in pop:
        fits.append(evaluate_ind(ind, base_ind, model, label_obj, lab_list))
    
    return fits


def load_pop(creator, base_ind, mean, stdev, n):
    individuals = []
    
    for i in range(n):
        # Generate mutations
        ind_array = np.random.normal(mean, stdev, 500)
        clipped_array = np.clip(ind_array, -1, 1)
        
        # Apply mutations to a certain number of genes
        mask = np.random.choice([True, False], size=500, p=[0.5, 0.5])
        clipped_array = np.where(mask, clipped_array, 0)
        
        individual = creator(clipped_array)
        individuals.append(individual)
        
    return individuals


# # Reverse mutation

def mutate_reverse(pop, mutrevpb, indrevpb):
    for i in range(len(pop)):
        if random.random() < mutrevpb:
            ind_array = np.copy(np.array(pop[i]))
            
            num_elem = int(indrevpb * len(ind_array))
            idx = np.random.choice(len(ind_array), num_elem, replace=False)
            ind_array[idx] = 0
            
            pop[i] = creator.Individual(ind_array)
    
    return pop


# # Statistics plotting

def print_fit_stats(fitnesses):
    mean = np.mean(fitnesses[:,0], axis=0)
    std = np.std(fitnesses[:,0], axis=0)
    print("-"*5, "Confidence fitness", "-"*5)
    print(f"Min: {min(fitnesses[:,0]):.5f}, Max: {max(fitnesses[:,0]):.5f}, Mean: {mean:.5f}, Std: {std:.5f}")
    
    mean = np.mean(fitnesses[:,1], axis=0)
    std = np.std(fitnesses[:,1], axis=0)
    print("-"*5, "Mutation magnitude fitness", "-"*5)
    print(f"Min: {min(fitnesses[:,1]):.5f}, Max: {max(fitnesses[:,1]):.5f}, Mean: {mean:.5f}, Std: {std:.5f}")
    
    mean = np.mean(fitnesses[:,2], axis=0)
    std = np.std(fitnesses[:,2], axis=0)
    print("-"*5, "Mutation number fitness", "-"*5)
    print(f"Min: {min(fitnesses[:,2]):.5f}, Max: {max(fitnesses[:,2]):.5f}, Mean: {mean:.5f}, Std: {std:.5f}")


def get_class_changes(base_ind, classifier, pop, lab_list):
    ind_copy = np.copy(np.array(base_ind))
    pred_base = classifier.predict(ind_copy.reshape(1,500))
    pred_label_base = lab_list[np.argmax(pred_base[0])]
    
    ind_mut = []
    for ind in pop:
        ind_mut.append(ind_copy[0] + ind)
    
    ind_mut = np.array(ind_mut)
    pred_mut = classifier.predict(ind_mut)
    
    changes = []
    for pred in pred_mut:
        pred_label_mut = lab_list[np.argmax(pred)]
        if pred_label_base != pred_label_mut:
            changes.append(1)
        else:
            changes.append(0)
            
    return changes


def get_class_changes_obj(base_ind, classifier, pop, lab_obj, lab_list):
    ind_copy = np.copy(np.array(base_ind))
    pred_base = classifier.predict(ind_copy.reshape(1,500))
    pred_label_base = lab_list[np.argmax(pred_base[0])]
    
    ind_mut = []
    for ind in pop:
        ind_mut.append(ind_copy[0] + ind)
    
    ind_mut = np.array(ind_mut)
    pred_mut = classifier.predict(ind_mut)
    
    changes = []
    for pred in pred_mut:
        pred_label_mut = lab_list[np.argmax(pred)]
        if pred_label_mut == lab_obj:
            changes.append(1)
        else:
            changes.append(0)
            
    return changes


# # Deap

# ## Configuration

def deap_configuration(latent_code):
    creator.create("Fitness", base.Fitness, weights=(1.0, -MUT_SIZE_WEIGHT, -MUT_NUM_WEIGHT))
    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=MUT_MEAN, sigma=MUT_STDEV, indpb=INDPB)
    toolbox.register("mutate_reverse", mutate_reverse)
    toolbox.register("select", tools.selSPEA2)
    toolbox.register("selectTournament", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_ind)
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox.register("population", load_pop, creator.Individual, latent_code[0],
                     mean=INI_MEAN, stdev=INI_STDEV)
    return creator, toolbox


# ## Generic algorithm

def spea2(base_ind, classifier, label_obj, lab_list,
          pop, toolbox, num_gens, sel_factor_pop, sel_factor_arch,
          mut_prob, mutrevpb, indrevpb, verb=0):
    archive = []
    miss_history = []
    curr_gen = 1

    while curr_gen <= num_gens:
        # Step 2 Fitness assignement
        fitnesses = evaluate_pop(pop, base_ind, classifier, label_obj, lab_list)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        class_ch = get_class_changes(base_ind, classifier, pop, lab_list)
        missclassified = np.mean(class_ch)
        miss_history.append(missclassified)
        
        if curr_gen%1 == 0 and verb==1:
            print("\nGEN: ", curr_gen)
            print_fit_stats(np.array(fitnesses))
            print('Percentage of targeted missclasification', missclassified, '%')

        # Step 3 Environmental selection
        archive  = toolbox.select(pop + archive, k=sel_factor_arch)

        # Step 4 Termination
        if curr_gen >= num_gens:
            return archive, miss_history

        # Step 5 Mating Selection
        mating_pool = toolbox.selectTournament(archive, k=sel_factor_pop)
        offspring_pool = list(map(toolbox.clone, mating_pool))

        # Step 6 Variation
        for child1, child2 in zip(offspring_pool[::2], offspring_pool[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

        for mutant in offspring_pool:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)

        offspring_pool = mutate_reverse(offspring_pool, mutrevpb, indrevpb)
        
        pop = offspring_pool
        curr_gen += 1
