import torch
import torch.nn as nn
import torch.optim as optim
import e3nn.o3
import e3nn.nn
from e3nn.nn import BatchNorm
from e3nn.o3 import FullyConnectedTensorProduct
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
import sklearn.metrics

# Generate pairs of vectors that are either parallel or perpendicular. (Binary or Dynamical Black Hole Formation)
def generate_parallel_perpendicular_data(n_samples=1000, noise_level=0.1): # Because nature is not perfect, add random noise to alter the numbers slightly
    v1_list = []
    v2_list = []
    labels = [] # parallel or perpendicular

    for i in range(n_samples): # create the list of vectors n times
        v1 = np.random.rand(3) # generate vector having 3 numbers (x, y, z)
        v1 = v1 / np.linalg.norm(v1) # normalize the vectors to a length of 1 (makes it fair for smaller numbers)
        
        label = np.random.randint(0, 2) # random 50/50 result of 0 or 1 (parallel or perpendicular)
        if label == 0:  # Parallel
            v2 = v1 + (np.random.randn(3) * noise_level)
        else:  # Perpendicular
            r = np.random.randn(3)
            v2 = r - np.dot(r, v1) * v1  # Gram-Schmidt orthogonalization
            v2 = v2 + (np.random.randn(3) * noise_level)
        v2 = v2 / np.linalg.norm(v2)

        # Apply same rotation to both vectors (preserves relationship)
        # This is the core of this project, as standard neural networks often struggle with geometric rotation
        rot_matrix = R.random().as_matrix() # generate a random 3x3 matrix
        v1_rotated = rot_matrix @ v1 # @ signifies matrix multiplication, this applies rotation to the vector)
        v2_rotated = rot_matrix @ v2
        v1_list.append(v1_rotated)
        v2_list.append(v2_rotated)
        labels.append(label)
        
    return { # creates an entry into a dictionary, the key being "v1", "v2", or "labels", and the values become PyTorch tensors
        "v1": torch.tensor(np.array(v1_list), dtype=torch.float32),
        "v2": torch.tensor(np.array(v2_list), dtype=torch.float32),
        "labels": torch.tensor(np.array(labels), dtype=torch.long)
    }

# This visualizes the vectors in a 3D space (Blue = parallel, red = perpendicular)
def visualize_vector_pairs(data, n_display=20):
    v1 = data["v1"].numpy() # Calls upon the tensor with a key. This also transforms it into a numpy array
    v2 = data["v2"].numpy()
    labels = data["labels"].numpy()

    fig = plt.figure(figsize=(10, 8)) # Create matplotlib figure, 10in wide and 8in tall
    ax = fig.add_subplot(111, projection='3d') # Creates a 3D subplot of 1 row, 1 column, 1 dimension (111)
    
    ax.scatter([0], [0], [0], c="black", marker="o", s=100) # This draws a single point at the coordinates (0,0,0) This acts as the center where the vectors come from.
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    for i in range(min(n_display, len(labels))): # iterates through n_display samples (20) and plots the vectors.
        color = "blue" if labels[i] == 0 else "red"
        # ax.quiver is what draws the vectors
        ax.quiver(0, 0, 0, v1[i, 0], v1[i, 1], v1[i, 2], # points to position x,y,z ([i,0], [i,1], [i,2])
                  color="gray", alpha=0.4, arrow_length_ratio=0.1)
        # Draw v2 in "color"
        ax.quiver(0, 0, 0, v2[i, 0], v2[i, 1], v2[i, 2],
                  color=color, alpha=0.8, arrow_length_ratio=0.1, linewidth=2)
    
    ax.set_xlim([-1, 1]) # limits the three axis to the range of -1 to 1
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title("Vector Pairs (Blue=Parallel, Red=Perpendicular)")
    plt.tight_layout() # auto adjusts the subplot parameters to minimize overlap
    

class AlignmentClassifier(nn.Module): # Creating the custom neural network model
    def __init__(self): # intialier; automatically executes when a new instance of the class is created
        super().__init__() # ensures any attributes defined in paren'ts class are set up for the child class instance
        self.irreps_in = e3nn.o3.Irreps("1o")  # Create's instance of the Irreps class (Irreducible Representations). Specifies how tensors transform under 3D rotations and reflections. "1o" = 1(vector) o(odd parity, flips its sign when inverted)
        self.irreps_output = e3nn.o3.Irreps("10x0e")  # "10x0e" = 10x(channels)0(scalar)e(even parity)
        
        # Tensor product: v1 ⊗ v2
        # Outputs geometric features encoding dot product, cross product, etc.
        self.tp = e3nn.o3.FullyConnectedTensorProduct( # FullyConnectedTensorProduct preforms tensor product between two input features to produce an output tensor which desired features
            self.irreps_in,  # vector v1
            self.irreps_in,  # vector v2
            self.irreps_output
        )
        self.norm = e3nn.nn.BatchNorm(self.irreps_output) # Implements batch normalization (normalizes based on norm/magnitude) Maintains symmetry properties

        # MLP classifier on scalar features
        self.classifier = nn.Sequential( # nn.sequential is a container class that stacks modules sequentially. Simple
            nn.Linear(10, 16), # Input Layer: input size of 10, with a linear transformation to 16 output features (hidden layer)
            nn.ReLU(), # activates Rectified Linear Unit non-linearity to the output of hidden layer
            nn.Linear(16, 2) # Output Layer: Final linear transformation to 2 output features, (parallel or perpendicular)
        )
# forward pass of a PyTorch neural network, processes through the layers to desired output
    def forward(self, v1, v2): # Executes computation of the model using two inputs, v1 and v2
        geometric_features = self.tp(v1, v2) # v1 and v2 are passed through self.tp (e3nn.o3.FullyConnectedTensorProduct)
        normalized_features = self.norm(geometric_features) # The stored variable geometric_features is then pushed through self.norm
        logits = self.classifier(normalized_features) # The output, raw unnormalized prediction scores for each possible output
        return logits # returns the raw scores (to be used in CrossEntropyLoss)

# the ML tool to train the alignment classifier
def train_model(model, data, n_epochs=100, lr=0.01): # model = the ML model needing training, data = dataset used for training, n_epochs = the amount of times the model will pass through the dataset, lr = the learning rate, a hyperparameter determining step size at each iteration
    optimizer = optim.Adam(model.parameters(), lr=lr) # initializes optimizer (Adam algorithm), adjusting model parameters (weights and biases) to reduce error
    criterion = nn.CrossEntropyLoss() # defines the loss function (criterion), used to measure difference between predictions and the answer. 
    v1 = data["v1"]
    v2 = data["v2"]
    labels = data["labels"]

    losses = [] # creates empty lists to store loss and accuracy values during each epoch 
    accuracies = []

    print(f"Training model for {n_epochs} epochs...")

    # loops through the dataset through n_epochs times. 
    for epoch in range(n_epochs):
        optimizer.zero_grad() # cleares the gradients from previous batch so there is no accumulation. "reset cache"
        logits = model(v1, v2) # forward pass, passes v1 and v2 through the model to get the raw scores.
        loss = criterion(logits, labels) # Sends logits and labels through the criterion (difference between predictions and answers)
        loss.backward() # Backpropagation, calculates the gradient of the loss in accordinace to every parameter. Figures out how much to change after each epoch
        optimizer.step() # Adjusts the parameters in the direction that minimizes loss using the optmizer (Adam algorithm)

        preds = torch.argmax(logits, dim=1) # Converts logits (raw scores) into predicted class labels by selecting the one with the highest score
        acc = (preds == labels).float().mean() # Calculates accuracy for current batch by checking which predictions == the answers, converts to float and then finds the mean

        losses.append(loss.item()) # Stores current loss value
        accuracies.append(acc.item()) # stores current accuracy value

        if epoch % 10 == 0: # checks if epoch is a multiple of 10
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {acc.item():.2f}") # prints every 10 epochs to monitor progress

    return losses, accuracies

# Tests to find if the ML model is truly equivariant 
def test_equivariance(model, n_tests=100): # tests the input model 100 times
   
    print(f"EQUIVARIANCE TEST")
    model.eval() # sets to PyTorch inference mode
    
    correct = 0 # intializes correct and total to 0 
    total = 0
    
    with torch.no_grad(): # loops n_test times without gradients being calculated
        for _ in range(n_tests):
            # Generate a pair of random vectors
            v1 = np.random.randn(3)
            v1 = v1 / np.linalg.norm(v1)
            
            label = np.random.randint(0, 2)
            if label == 0:  # Parallel
                v2 = v1 + 0.1 * np.random.randn(3)
            else:  # Perpendicular
                r = np.random.randn(3)
                v2 = r - np.dot(r, v1) * v1
            v2 = v2 / np.linalg.norm(v2)
            
            # Rotate BOTH vectors by the SAME rotation (preserves relationship)
            rot = R.random().as_matrix()
            v1_rot = rot @ v1
            v2_rot = rot @ v2
            
            # Predict
            v1_t = torch.tensor(v1_rot, dtype=torch.float32).unsqueeze(0) # converts existing data in v1_rot into a PyTorch tensor with a 32-bit floating point. .unsqueeze(0) adds a new dimension of size 1 at index 0 (batch size)
            v2_t = torch.tensor(v2_rot, dtype=torch.float32).unsqueeze(0)
            pred = torch.argmax(model(v1_t, v2_t)).item() # passes the two tensors through the ML model to get an output, argmax finds index with max value (label), .item() extracts the value of the predicted label from the tensor
            
            if pred == label: # conditional to compare the prediction to the answer
                correct += 1
            total += 1
    
    accuracy = correct / total # calculates the accuracy 
    print(f"Accuracy on rotated pairs (same rotation): {accuracy*100:.1f}%")
    return accuracy

# tests on a different validation set
def test_on_validation_set(model, val_data):
    print(f"Valdidation Test Set")
    model.eval()
    
    with torch.no_grad(): # disables gradient calculation
        logits = model(val_data["v1"], val_data["v2"]) # forward pass, v1 and v2 are passed through the ML model
        preds = torch.argmax(logits, dim=1) # finds max valued index (label), converts logits into predictions
        accuracy = (preds == val_data["labels"]).float().mean().item() # calculates accuracy for the predictions that = the label of val_data, converted to a float and calculating the mean
    
    print(f"Validation Accuracy: {accuracy*100:.1f}%")
    return accuracy


if __name__ == "__main__":
    print("Equivariant Vector Alignment Classifier")
    
    # Generate data
    print("Generating training data")
    train_data = generate_parallel_perpendicular_data(n_samples=2000, noise_level=0.1)
    
    print("Generating validation data")
    val_data = generate_parallel_perpendicular_data(n_samples=500, noise_level=0.1)
    
    # Visualize examples
    print("Visualizing sample vector pairs")
    visualize_vector_pairs(train_data, n_display=20)
    
    # Initialize and train model
    print("Initializing Equivariant Alignment Classifier")
    model = AlignmentClassifier()
    
    losses, accs = train_model(model, train_data, n_epochs=100, lr=0.01) # calls the train_model function (which returns losses and accuracies)
    print(f"Final Training Accuracy: {accs[-1]*100:.1f}%") # grabs the accuracy from the last epoch (-1), multiplying by 100 to get a %
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(accs)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Run tests
    val_acc = test_on_validation_set(model, val_data)
    eq_acc = test_equivariance(model, n_tests=200)
    
    print("Final Results")
    print(f"Training Accuracy:   {accs[-1]*100:.1f}%")
    print(f"Validation Accuracy: {val_acc*100:.1f}%")
    print(f"Equivariance Test:   {eq_acc*100:.1f}%")
    
    plt.show()