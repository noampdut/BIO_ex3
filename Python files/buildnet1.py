import numpy as np
import sys

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 150
MUTATION_RATE = 0.05

# Neural Network Parameters
INPUT_SIZE = 16
HIDDEN_SIZE = 8
OUTPUT_SIZE = 1

'''
This function calculate sigmoid activation function
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
This function create random possible solutions represents model weights
'''
def initialize_population():
    return np.random.uniform(-1, 1, size=(POPULATION_SIZE, (INPUT_SIZE + 1) * HIDDEN_SIZE + (HIDDEN_SIZE + 1) * OUTPUT_SIZE))


'''
This function get possible solution and parse it to weights
'''
def decode_chromosome(chromosome):
    w1 = chromosome[:INPUT_SIZE * HIDDEN_SIZE].reshape((INPUT_SIZE, HIDDEN_SIZE))
    b1 = chromosome[INPUT_SIZE * HIDDEN_SIZE:(INPUT_SIZE + 1) * HIDDEN_SIZE]
    w2 = chromosome[(INPUT_SIZE + 1) * HIDDEN_SIZE:(INPUT_SIZE + 1) * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE].reshape((HIDDEN_SIZE, OUTPUT_SIZE))
    return w1, b1, w2


'''
This function get chromosome- possible solution and the train data
The function return the fitness score of the chromosome by calculation 
the percentage of accuracy of the model with the chromosome weights
'''
def evaluate_fitness(chromosome, data):
    w1, b1, w2 = decode_chromosome(chromosome)
    correct_count = 0
    for x, y in data:
        hidden = sigmoid(np.dot(x, w1) + b1)
        output = sigmoid(np.dot(hidden, w2))
        prediction = int(output[0] > 0.5)
        if prediction == y:
            correct_count += 1
    return correct_count / len(data)


'''
This function get the array of fitness scores of the population and return random two solutions 
according to the distribution of the fitness scores
'''
def select_parents(fitness_scores):
    probabilities = fitness_scores / np.sum(fitness_scores)
    return np.random.choice(POPULATION_SIZE // 2, size=2, replace=False, p=probabilities)


'''
This function get two solutions and do the crossover action and return the result
'''
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


'''
This function get possible solution- chromosome and make a mutation in it with the probability of MUTATION_RATE
'''
def mutate(chromosome):
    for i in range(len(chromosome)):
        if np.random.random() < MUTATION_RATE:
            chromosome[i] = np.random.uniform(-1, 1)
    return chromosome


'''
This function get the population and the train data
calculate the fitness score of the population and return half of the population with the highest scores
'''
def selection(population, data):
    sorted_population = sorted(population, key=lambda key: evaluate_fitness(key, data), reverse=True)
    selected = sorted_population[:int(POPULATION_SIZE / 2)]
    return selected

'''
This function is the main loop of the genetic algorithm.
It loop until reach NUM_GENERATIONS and in each generation take the top half of the population 
(possible solution - model weights) 
and regenerate using cross and mutation functions the other half.
Return the best solution he found according to fitness function.
'''
def genetic_algorithm(data):
    population = initialize_population()
    for i in range(NUM_GENERATIONS):
        # print progress
        print(str(i+1) + " / " + str(NUM_GENERATIONS))
        selected = selection(population, data)
        new_population = selected[:]

        fitness_scores = np.array([evaluate_fitness(chromosome, data) for chromosome in new_population])

        for _ in range(POPULATION_SIZE // 4):
            parents_indices = select_parents(fitness_scores)
            parents = population[parents_indices]
            child1, child2 = crossover(parents[0], parents[1])
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = np.array(new_population)

    fitness_scores = np.array([evaluate_fitness(chromosome, data) for chromosome in population])

    best_chromosome = population[np.argmax(fitness_scores)]
    return decode_chromosome(best_chromosome)


'''
This function get the weights and the test data and return the accuracy of model with the given weights 
'''
def test(w1, b1, w2, test_data):
    correct_count = 0
    for x, y in test_data:
        hidden = sigmoid(np.dot(x, w1) + b1)
        output = sigmoid(np.dot(hidden, w2))
        prediction = int(output[0] > 0.5)
        if prediction == y:
            correct_count += 1
    return correct_count / len(test_data)


# Main program
if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # Read data from the train file
    data = []
    with open(train_file, "r") as file:
        for line in file:
            line = line.strip()
            bit_string, classification = line.split()
            bit_string = [int(bit) for bit in bit_string]
            classification = int(classification)
            data.append((bit_string, classification))

    X_train = [d[0] for d in data]  # Input features
    y_train = [d[1] for d in data]  # Class labels

    # Read data from the test file
    data = []
    with open(test_file, "r") as file:
        for line in file:
            line = line.strip()
            bit_string, classification = line.split()
            bit_string = [int(bit) for bit in bit_string]
            classification = int(classification)
            data.append((bit_string, classification))

    # Split data into training and testing sets
    X_test = [d[0] for d in data]  # Input features
    y_test = [d[1] for d in data]  # Class labels

    # Convert data to NumPy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Combine X_train and y_train into a single variable
    train_data = list(zip(X_train, y_train))

    # Combine X_test and y_test into a single variable
    test_data = list(zip(X_test, y_test))

    # Run the genetic algorithm
    best_weights = genetic_algorithm(train_data)

    # Display the weights
    w1, b1, w2 = best_weights

    # Save weights to a file
    with open("wnet1.txt", "w") as file:
        file.write("Hidden Layer Weights (w1):\n")
        np.savetxt(file, w1)
        file.write("\nHidden Layer Biases (b1):\n")
        np.savetxt(file, b1)
        file.write("\nOutput Layer Weights (w2):\n")
        np.savetxt(file, w2)

    print("Test Data Accuracy:")
    print(test(w1, b1, w2, test_data))
