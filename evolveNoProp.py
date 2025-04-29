import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Tuple, Callable, Optional

# Constants for the evolutionary algorithm
POPULATION_SIZE = 100
GENERATIONS = 150
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_COUNT = 10  # Keep the top N individuals across generations

# Device to use (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------
# 1. Define the Classifier Model (Evolvable)
# --------------------------------------------------------------------------------

class SimpleClassifier(nn.Module):
    """
    A simple convolutional neural network for binary classification, designed to be evolved.
    The architecture is fixed, but the weights are evolved.  Handles the specified
    input dimensions (1, 1, 16, 214, 214).
    """
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        # Calculate the size of the flattened features after the convolutional layers.
        # Hardcoded based on the input size and pooling.
        self.fc1_input_size = 16 * 2 * 2 * 26 * 26  #  Adjusted for 3D pooling
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, 1)  # Output a single value for binary classification.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        Args:
            x: Input tensor of shape (batch_size, 1, 16, 214, 214).
        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_size)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output a single value
        return x

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
                offspring_weights[i].data = (mask * parent1_weights[i].data) + ((1 - mask) * parent2_weights[i].data)
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
    """
    Evolve a population of SimpleClassifier models to perform binary classification
    on the given training data.  Includes a validation set for monitoring
    performance and early stopping.
    Args:
        train_loader: DataLoader for the training data.
        val_loader:   DataLoader for the validation data.
    Returns:
        The best evolved SimpleClassifier model.
    """
    # 1. Initialize the population
    population = [SimpleClassifier().to(DEVICE) for _ in range(POPULATION_SIZE)]
    best_individual: Optional[SimpleClassifier] = None # type: ignore
    best_fitness = -float('inf')
    # Store the best fitness for each generation
    best_fitnesses = []
    # 2. Main evolutionary loop
    for generation in range(GENERATIONS):
        # 2.1 Evaluate the fitness of each individual in the population
        fitnesses = [evaluate_fitness(individual, train_loader) for individual in population]
        # 2.2 Print the best fitness in the current generation
        best_fitness_gen = max(fitnesses)
        best_fitnesses.append(best_fitness_gen)
        print(f"Generation {generation + 1}/{GENERATIONS} - Best Fitness: {best_fitness_gen:.4f}")

        # 2.3 Update the best individual found so far
        if best_fitness_gen > best_fitness:
            best_fitness = best_fitness_gen
            best_individual = population[fitnesses.index(best_fitness_gen)].clone()

        # 2.4 Perform selection, crossover, and mutation to create the next generation
        parents = select_parents(population, fitnesses)
        offspring = []
        for parent1, parent2 in parents:
            child = crossover(parent1, parent2, CROSSOVER_RATE)
            child = mutate(child, MUTATION_RATE)
            offspring.append(child.to(DEVICE))

        # 2.5 Elitism: Keep the top ELITISM_COUNT individuals from the previous generation
        # Sort the population by fitness (descending order)
        sorted_population = [p for _, p in sorted(zip(fitnesses, population), reverse=True)]
        elites = sorted_population[:ELITISM_COUNT]
        # Replace the least fit individuals in the offspring population with the elites
        offspring[-ELITISM_COUNT:] = elites

        population = offspring

        # 2.6 Evaluate the best individual on the validation set.
        if val_loader:
            val_fitness = evaluate_fitness(best_individual, val_loader)
            print(f"  Validation Fitness: {val_fitness:.4f}")
    print("Evolutionary process complete.")
    if best_individual is None:
        raise ValueError("No best individual found during evolution.")
    return best_individual



# --------------------------------------------------------------------------------
# 5. Main Script (Example Usage)
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # 5.1. Create a dummy dataset and data loaders (replace with your actual data)
    # Example data: (batch_size, 1, 16, 214, 214)
    train_data = torch.randn(100, 1, 16, 214, 214)  # 100 samples
    train_labels = torch.randint(0, 2, (100, 1)).float()  # Binary labels (0 or 1)
    val_data = torch.randn(50, 1, 16, 214, 214)
    val_labels = torch.randint(0, 2, (50, 1)).float()

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)

    # 5.2 Evolve the population and get the best model
    best_model = evolve_population(train_loader, val_loader)

    # 5.3 Evaluate the best model on the test set (replace with your test data)
    test_data = torch.randn(20, 1, 16, 214, 214)  # 20 samples
    test_labels = torch.randint(0, 2, (20, 1)).float()
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    test_accuracy = evaluate_fitness(best_model, test_loader)
    print(f"Test Accuracy of the best evolved model: {test_accuracy:.4f}")

    # 5.4  Save the best model (optional)
    torch.save(best_model.state_dict(), "best_evolved_model.pth")
    print("Best model saved to best_evolved_model.pth")