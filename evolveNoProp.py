import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Tuple, Callable, Optional

# Constants for the evolutionary algorithm
POPULATION_SIZE = 150
GENERATIONS = 1000
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 10  # Keep the top N individuals across generations
TOURNAMENT_SIZE = 8  # Number of individuals to select for tournament selection
# Device to use (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------
# 1. Define the Classifier Model (Evolvable)
# --------------------------------------------------------------------------------



class SimpleClassifier(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(SimpleClassifier, self).__init__()
        # Input shape: (batch_size, 1, 16, 214, 214)
        
        # First block: 1 -> 16
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))
        # Size after pool1: (16, 8, 107, 107)
        
        # Second block: 16 -> 32
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))
        # Size after pool2: (32, 4, 54, 54)
        
        # Third block: 32 -> 64
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))
        # Size after pool3: (64, 2, 27, 27)
        
        # Fourth block: 64 -> 128
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))
        # Size after pool4: (128, 1, 14, 14)
        
        # Fifth block: 128 -> 256
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        # Size after pool5: (256, 1, 7, 7)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout3d(0.3)
        
        # Calculate the actual output size using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 16, 214, 214)
            x = self.forward_features(dummy_input)
            self.fc1_input_size = x.view(1, -1).size(1)
            # print(f"Calculated input size for fc1: {self.fc1_input_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward_features(self, x):
        # Add shape tracking for debugging
        # print(f"Input shape: {x.shape}")
        
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        # print(f"After block 1: {x.shape}")
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        # print(f"After block 2: {x.shape}")
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        # print(f"After block 3: {x.shape}")
        
        # Fourth block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        # print(f"After block 4: {x.shape}")
        
        # Fifth block
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        # print(f"After block 5: {x.shape}")
        
        return x
    
    def forward(self, x):
        # Feature extraction
        x = self.forward_features(x)
        
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def clone(self) -> 'SimpleClassifier':
        """
        Create a deep copy of the model with proper initialization
        """
        new_model = SimpleClassifier()
        new_model.load_state_dict(self.state_dict())
        return new_model



    def get_weights(self) -> List[torch.Tensor]:
        """
        Get the weights of the model as a list of tensors.  Crucial for evolution.
        Returns:
            A list of the model's weights (tensors).
        """
        return [param.data for param in self.parameters()]

    def set_weights(self, weights: List[torch.Tensor]) -> None:
        """
        Set the weights of the model from a list of tensors.  Crucial for evolution.
        Args:
            weights: A list of tensors representing the new weights.
        Raises:
            ValueError: If the number of weights provided does not match the
                number of weights in the model.
        """
        if len(weights) != len(list(self.parameters())):
            raise ValueError("Number of weights does not match the model's parameters.")
        for param, weight in zip(self.parameters(), weights):
            param.data.copy_(weight.data)  # Copy the data, not the tensor itself.

    def clone(self) -> 'SimpleClassifier':
        """
        Create a deep copy of the model.  This is essential for maintaining
        independent individuals in the population.
        Returns:
            A new SimpleClassifier object with the same architecture and weights.
        """
        # Create a new instance of the model
        new_model = SimpleClassifier()
        new_model.to(self.device)  # Move to the same device as the original model
        # Copy the weights from the current model to the new model
        new_model.load_state_dict(self.state_dict())
        return new_model



# --------------------------------------------------------------------------------
# 2. Data Handling and Fitness Evaluation
# --------------------------------------------------------------------------------

def binary_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the binary accuracy of the predictions.
    Args:
        outputs:  Tensor of shape (batch_size, 1) containing the model's output.
        labels:   Tensor of shape (batch_size, 1) containing the true labels (0 or 1).
    Returns:
        The binary accuracy as a float.
    """
    # Ensure labels are 0 or 1
    labels = labels.float()
    outputs = outputs.float()
    # Apply sigmoid to the outputs to get probabilities
    probs = torch.sigmoid(outputs)
    # Threshold probabilities to get predictions (0 or 1)
    predictions = (probs >= 0.5).float()
    # Calculate the number of correct predictions
    correct_predictions = (predictions == labels).sum().item()
    # Calculate the total number of predictions
    total_predictions = labels.size(0)
    # Return the accuracy
    return correct_predictions / total_predictions

def evaluate_fitness(model: SimpleClassifier,
                    data_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the fitness of a model on the given data loader (using binary accuracy).
    Args:
        model: The SimpleClassifier model to evaluate.
        data_loader:  DataLoader providing the data and labels.
    Returns:
        The fitness (binary accuracy) of the model.
    """
    model.eval()  # Set the model to evaluation mode
    total_accuracy = 0.0
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(DEVICE)  # Move data to the device
            batch_labels = batch_labels.to(DEVICE)
            # Forward pass: get the model's output
            outputs = model(batch_data)
            # Calculate the accuracy for the current batch
            batch_accuracy = binary_accuracy(outputs, batch_labels)
            total_accuracy += batch_accuracy
    # Return the average accuracy over all batches
    return total_accuracy / len(data_loader)



# --------------------------------------------------------------------------------
# 3. Genetic Operators
# --------------------------------------------------------------------------------

def mutate(individual: SimpleClassifier, mutation_rate: float) -> SimpleClassifier:
    """
    Mutate the weights of an individual (a SimpleClassifier model).
    Args:
        individual: The SimpleClassifier model to mutate.
        mutation_rate: The probability of mutating each weight.
    Returns:
        A new SimpleClassifier model with mutated weights.
    """
    mutated_individual = individual.clone() #important
    with torch.no_grad():
        for param in mutated_individual.parameters():
            # Create a random tensor of the same shape as the parameter
            random_tensor = torch.rand(param.size()).to(DEVICE)
            # Create a mask for the weights that will be mutated
            mutation_mask = (random_tensor < mutation_rate).float()
            # Generate the random values to add to the weights
            mutation_values = torch.randn(param.size()).to(DEVICE) * mutation_mask
            # Add the mutation values to the parameters
            param.data += mutation_values
    return mutated_individual



def crossover(parent1: SimpleClassifier, parent2: SimpleClassifier, crossover_rate: float) -> SimpleClassifier:
    """
    Perform crossover between two parent individuals to create a new offspring individual.
    This version correctly handles the case where the parents have different numbers
    of weight tensors (which should not happen with our fixed architecture, but
    is good practice for more general code).
    Args:
        parent1: The first parent SimpleClassifier model.
        parent2: The second parent SimpleClassifier model.
        crossover_rate: The probability of performing crossover for each weight.
    Returns:
        A new SimpleClassifier model (the offspring).
    """
    offspring = parent1.clone() #important
    parent1_weights = list(parent1.parameters())
    parent2_weights = list(parent2.parameters())
    offspring_weights = list(offspring.parameters())

    # Iterate through the weights of the offspring.
    for i in range(len(offspring_weights)):
        # Check if crossover should be applied for this weight tensor.
        if torch.rand(1).item() < crossover_rate:
            # Ensure that the parent tensors have the same shape.
            if parent1_weights[i].shape == parent2_weights[i].shape:
                # Create a mask to determine which parts of the weight tensor
                # come from parent1 and which come from parent2.
                mask = torch.rand(parent1_weights[i].size()).to(DEVICE) < 0.5
                # Create the offspring weight tensor by combining the weights
                # from the parents using the mask.
                offspring_weights[i].data = (mask * parent1_weights[i].data) + ((~mask) * parent2_weights[i].data)
            else:
                # If the parents have different shapes, it's probably an error,
                # but we'll just use the weights from parent1.  A warning
                # could also be printed.
                offspring_weights[i].data = parent1_weights[i].data
        else:
            # If no crossover, use the weights from parent1.
            offspring_weights[i].data = parent1_weights[i].data
    return offspring



def select_parents(population: List[SimpleClassifier],
                    fitnesses: List[float]) -> List[Tuple[SimpleClassifier, SimpleClassifier]]:
    """
    Select pairs of parents for crossover using roulette wheel selection.
    Args:
        population: The current population of SimpleClassifier models.
        fitnesses:  A list of fitness values corresponding to the individuals
                    in the population.
    Returns:
        A list of tuples, where each tuple contains two parent individuals.
    """
    # Ensure that fitnesses are non-negative. If there are negative fitness
    # values, shift them to be positive.
    min_fitness = min(fitnesses)
    if min_fitness < 0:
        adjusted_fitnesses = [f - min_fitness for f in fitnesses]
    else:
        adjusted_fitnesses = fitnesses

    # Calculate the total fitness
    total_fitness = sum(adjusted_fitnesses)
    # Calculate the selection probabilities for each individual
    probabilities = [f / total_fitness for f in adjusted_fitnesses]
    # Select pairs of parents
    parents = []
    for _ in range(len(population) // 2):  # Select half the population size as pairs
        # Select two parents using the calculated probabilities
        parent1_index = np.random.choice(len(population), p=probabilities)
        parent2_index = np.random.choice(len(population), p=probabilities)
        parents.append((population[parent1_index], population[parent2_index]))
    return parents



# --------------------------------------------------------------------------------
# 4. Main Evolutionary Algorithm Loop
# --------------------------------------------------------------------------------


def evolve_population(train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader) -> SimpleClassifier:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    generationFitnesses = []
    
    # Initialize population
    population = [SimpleClassifier().to(device) for _ in range(POPULATION_SIZE)]
    best_individual = None
    best_fitness = float('-inf')
    
    for generation in range(GENERATIONS):
        # Evaluate fitness for each individual
        fitnesses = [evaluate_fitness(individual, train_loader) for individual in population]
        
        # Create list of (fitness, model) tuples and sort by fitness
        population_with_fitness = list(zip(fitnesses, population))
        # Sort based on fitness values (first element of tuple)
        population_with_fitness.sort(key=lambda x: x[0], reverse=True)
        
        # Unzip the sorted population
        fitnesses, population = zip(*population_with_fitness)
        population = list(population)  # Convert tuple back to list
        
        # Update best individual
        if fitnesses[0] > best_fitness:
            best_fitness = fitnesses[0]
            best_individual = population[0].clone()
            # Save the best individual state dict
            if best_individual is not None:
                # Ensure the best individual is on the correct device
                best_individual = best_individual.to(device)
                # Save the state dict of the best individual
                if generation % 10 == 0:  # Save every 10 generations
                    print(f"Saving best model at generation {generation + 1}")
                    torch.save(best_individual.state_dict(), f"best_model_generation_{generation + 1}.pth")


        generationFitnesses.append(best_fitness)
        
        print(f"Generation {generation + 1}/{GENERATIONS} - Best Fitness: {best_fitness:.4f}")
        
        # Selection
        parents = []
        for _ in range(POPULATION_SIZE // 2):
            tournament = random.sample(list(enumerate(fitnesses)), TOURNAMENT_SIZE)
            winner_idx = max(tournament, key=lambda x: x[1])[0]
            parents.append(population[winner_idx])
        
        # Create pairs of parents
        random.shuffle(parents)
        parent_pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
        
        # Create new population through crossover and mutation
        new_population = []
        for parent1, parent2 in parent_pairs:
            # Create two children from each pair of parents
            for _ in range(2):
                child = crossover(parent1, parent2, CROSSOVER_RATE)
                child = mutate(child, MUTATION_RATE)
                new_population.append(child)
        
        # Ensure all models in the new population are on the correct device
        population = [model.to(device) for model in new_population]
        
        # Optional: Add elitism by replacing the worst individual with the best from previous generation
        if best_individual is not None:
            population[-1] = best_individual.clone()
            
    #save the array of fitness per generation to a csv
    generationFitnesses = np.array(generationFitnesses)
    generationFitnesses = generationFitnesses.reshape(-1, 1)
    np.savetxt("generation_fitnesses.csv", generationFitnesses, delimiter=",")
    return best_individual







