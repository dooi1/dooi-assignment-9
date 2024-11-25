import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Xavier initialization for weights
        limit1 = np.sqrt(6 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        limit2 = np.sqrt(6 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

        # For storing activations and gradients
        self.z1 = None  # Input to activation function in hidden layer
        self.a1 = None  # Output from activation function in hidden layer
        self.z2 = None  # Input to activation function in output layer
        self.a2 = None  # Output from activation function in output layer

        # For visualization
        self.grads = {}

    def forward(self, X):
        # Forward pass
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        # Activation function in hidden layer
        if self.activation_fn == 'tanh':
            self.a1 = np.tanh(self.z1)
        elif self.activation_fn == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation_fn == 'sigmoid':
            self.a1 = 1 / (1 + np.exp(-self.z1))
        else:
            raise ValueError('Unknown activation function')

        # Hidden layer to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Output activation (sigmoid for binary classification)
        self.a2 = 1 / (1 + np.exp(-self.z2))

        # Store activations for visualization
        return self.a2

    def backward(self, X, y):
        m = y.shape[0]  # Number of examples

        # Compute delta for output layer
        delta2 = (self.a2 - y)  # Cross-entropy loss derivative

        # Gradients for W2 and b2
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # Backpropagate to hidden layer
        if self.activation_fn == 'tanh':
            da1 = np.dot(delta2, self.W2.T)
            dz1 = da1 * (1 - np.power(self.a1, 2))
        elif self.activation_fn == 'relu':
            da1 = np.dot(delta2, self.W2.T)
            dz1 = da1.copy()
            dz1[self.z1 <= 0] = 0
        elif self.activation_fn == 'sigmoid':
            da1 = np.dot(delta2, self.W2.T)
            sig_z1 = 1 / (1 + np.exp(-self.z1))
            dz1 = da1 * sig_z1 * (1 - sig_z1)
        else:
            raise ValueError('Unknown activation function')

        # Gradients for W1 and b1
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Store gradients for visualization
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1
        self.grads['dW2'] = dW2
        self.grads['db2'] = db2

    def compute_loss(self, y):
        # Compute loss (cross-entropy)
        m = y.shape[0]
        loss = -np.mean(y * np.log(self.a2 + 1e-8) + (1 - y) * np.log(1 - self.a2 + 1e-8))
        return loss

def generate_data(n_samples=200):
    np.random.seed(0)
    # Generate input data
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Compute loss
    loss = mlp.compute_loss(y)
    print(f"Step {frame * 10}, Loss: {loss:.4f}")

    # Plot hidden features
    hidden_features = mlp.a1  # Shape (n_samples, 3)
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )
    ax_hidden.set_title(f'Hidden Space at Step {frame * 10}')
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')

    # Hyperplane visualization in the hidden space
    W2 = mlp.W2.flatten()
    b2 = mlp.b2.item()
    if W2[2] != 0:
        xlim = ax_hidden.get_xlim()
        ylim = ax_hidden.get_ylim()
        x = np.linspace(xlim[0], xlim[1], 10)
        y_ = np.linspace(ylim[0], ylim[1], 10)
        X_grid, Y_grid = np.meshgrid(x, y_)
        Z_grid = (-W2[0]*X_grid - W2[1]*Y_grid - b2)/W2[2]
        ax_hidden.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, color='green')

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Forward pass through network
    z1 = np.dot(grid_points, mlp.W1) + mlp.b1
    if mlp.activation_fn == 'tanh':
        a1 = np.tanh(z1)
    elif mlp.activation_fn == 'relu':
        a1 = np.maximum(0, z1)
    elif mlp.activation_fn == 'sigmoid':
        a1 = 1 / (1 + np.exp(-z1))
    else:
        raise ValueError('Unknown activation function')
    z2 = np.dot(a1, mlp.W2) + mlp.b2
    a2 = 1 / (1 + np.exp(-z2))
    Z = a2.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, levels=50, cmap='bwr', alpha=0.6)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title(f'Input Space at Step {frame * 10}')
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # Visualize features and gradients
    layer_sizes = [2, 3, 1]
    layer_positions = [0, 1, 2]
    neuron_positions = []

    for i, size in enumerate(layer_sizes):
        y_positions = np.linspace(0.1, 0.9, size)
        x_positions = np.full(size, layer_positions[i])
        neuron_positions.append(list(zip(x_positions, y_positions)))

    flattened_positions = [pos for layer in neuron_positions for pos in layer]

    # Plot neurons
    for pos in flattened_positions:
        circle = Circle(pos, 0.03, color='white', ec='black', zorder=4)
        ax_gradient.add_patch(circle)

    # Plot edges with gradient magnitudes
    max_grad = max(
        np.abs(mlp.grads['dW1']).max(),
        np.abs(mlp.grads['dW2']).max()
    )

    for i, (x0, y0) in enumerate(neuron_positions[0]):
        for j, (x1, y1) in enumerate(neuron_positions[1]):
            grad = mlp.grads['dW1'][i, j]
            linewidth = (np.abs(grad) / max_grad) * 5  # Scale for visibility
            color = 'green' if grad > 0 else 'red'
            ax_gradient.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth)

    for i, (x0, y0) in enumerate(neuron_positions[1]):
        for j, (x1, y1) in enumerate(neuron_positions[2]):
            grad = mlp.grads['dW2'][i, j]
            linewidth = (np.abs(grad) / max_grad) * 5  # Scale for visibility
            color = 'green' if grad > 0 else 'red'
            ax_gradient.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth)

    ax_gradient.axis('off')
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.set_title(f'Gradients at Step {frame * 10}')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Adjust layouts
    plt.tight_layout()

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(
            update,
            mlp=mlp,
            ax_input=ax_input,
            ax_hidden=ax_hidden,
            ax_gradient=ax_gradient,
            X=X,
            y=y
        ),
        frames=step_num // 10,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
