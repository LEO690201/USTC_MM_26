"""
Main experiment script for Deep Ritz method.
Trains the model on Poisson equation and evaluates error.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from network import DeepRitzNetwork, DeepRitzNetworkNoResNet
from solver import DeepRitzSolver
from examples import create_example
import os


def run_experiment(example_name='square', use_resnet=True, 
                   use_random_sampling=True, num_epochs=2000,
                   batch_size_interior=256, batch_size_boundary=64,
                   hidden_dim=64, num_blocks=4):
    """
    Run Deep Ritz experiment.
    
    Args:
        example_name: 'square', 'square_inhom', or 'circle'
        use_resnet: Whether to use ResNet architecture
        use_random_sampling: Whether to use random sampling (True) or fixed (False)
        num_epochs: Number of training epochs
        batch_size_interior: Interior batch size
        batch_size_boundary: Boundary batch size
        hidden_dim: Hidden dimension of network
        num_blocks: Number of ResNet blocks
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create example
    example = create_example(example_name)
    print(f"\n{'='*60}")
    print(f"Example: {example.name}")
    print(f"Architecture: {'ResNet' if use_resnet else 'Simple FC'}")
    print(f"Sampling: {'Random' if use_random_sampling else 'Fixed'}")
    print(f"{'='*60}")
    
    domain_bounds = example.get_domain_bounds()
    
    # Create model
    if use_resnet:
        model = DeepRitzNetwork(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            output_dim=1
        )
    else:
        model = DeepRitzNetworkNoResNet(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_blocks,
            output_dim=1
        )
    
    # Create solver
    solver = DeepRitzSolver(
        model=model,
        domain_bounds=domain_bounds,
        boundary_points_func=example.boundary_points,
        f_func=example.f,
        g_func=example.g,
        penalty_beta=1000.0,
        device=device
    )
    
    # Train
    print(f"\nTraining for {num_epochs} epochs...")
    solver.train(
        num_epochs=num_epochs,
        batch_size_interior=batch_size_interior,
        batch_size_boundary=batch_size_boundary,
        learning_rate=0.001,
        verbose=True
    )
    
    # Generate test points for error evaluation
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    
    # Create grid for testing
    if 'circle' in example_name.lower():
        # For circle, sample interior points
        n_test = 100
        test_points_list = []
        for _ in range(n_test * 10):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            if x**2 + y**2 <= 1.0:
                test_points_list.append([x, y])
        test_points = torch.tensor(test_points_list, dtype=torch.float32)[:n_test]
    else:
        # For square, create regular grid
        n_side = 50
        x_test = np.linspace(x_min + 0.01, x_max - 0.01, n_side)
        y_test = np.linspace(y_min + 0.01, y_max - 0.01, n_side)
        xx, yy = np.meshgrid(x_test, y_test)
        test_points = torch.tensor(
            np.column_stack([xx.flatten(), yy.flatten()]),
            dtype=torch.float32
        )
    
    # Compute errors
    print("\nComputing errors...")
    l2_error = solver.compute_l2_error(test_points, example.u_exact)
    linf_error = solver.compute_linf_error(test_points, example.u_exact)
    
    print(f"L2 Error:  {l2_error:.6e}")
    print(f"L∞ Error:  {linf_error:.6e}")
    
    # Save results
    result = {
        'example': example_name,
        'use_resnet': use_resnet,
        'use_random': use_random_sampling,
        'num_epochs': num_epochs,
        'l2_error': l2_error,
        'linf_error': linf_error,
        'loss_history': solver.loss_history,
        'final_loss': solver.loss_history[-1] if solver.loss_history else None
    }
    
    return solver, example, test_points, result


def plot_results(solver, example, test_points, result):
    """Plot training loss and predictions."""
    
    os.makedirs('results/figures', exist_ok=True)
    
    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(result['loss_history'], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f"Training Loss - {example.name} "
                 f"({('ResNet' if result['use_resnet'] else 'No ResNet')})",
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    arch = 'resnet' if result['use_resnet'] else 'no_resnet'
    plt.savefig(f'results/figures/loss_{result["example"]}_{arch}.png', dpi=150)
    print(f"Saved: results/figures/loss_{result['example']}_{arch}.png")
    plt.close()
    
    # Predictions vs Exact
    device = solver.device
    test_points_device = test_points.to(device)
    
    with torch.no_grad():
        u_pred = solver.model(test_points_device).cpu().numpy()
    
    u_exact = np.array([
        example.u_exact(test_points[i, 0].item(), test_points[i, 1].item())
        for i in range(test_points.shape[0])
    ])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Predicted solution
    if 'circle' in example.name.lower():
        # For circle
        domain_bounds = example.get_domain_bounds()
        x_min, x_max = domain_bounds[0]
        y_min, y_max = domain_bounds[1]
        
        n_grid = 50
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        grid_points = torch.tensor(
            np.column_stack([xx.flatten(), yy.flatten()]),
            dtype=torch.float32
        ).to(device)
        
        with torch.no_grad():
            u_grid_pred = solver.model(grid_points).cpu().numpy().reshape(xx.shape)
        
        mask = xx**2 + yy**2 > 1.0
        u_grid_pred[mask] = np.nan
        
        im0 = axes[0].contourf(xx, yy, u_grid_pred, levels=20, cmap='viridis')
        axes[0].set_title('Predicted Solution', fontsize=12)
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0])
        
        u_grid_exact = example.u_exact(xx, yy)
        u_grid_exact[mask] = np.nan
        
        im1 = axes[1].contourf(xx, yy, u_grid_exact, levels=20, cmap='viridis')
        axes[1].set_title('Exact Solution', fontsize=12)
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1])
        
        error_grid = np.abs(u_grid_pred - u_grid_exact)
        error_grid[mask] = np.nan
        
        im2 = axes[2].contourf(xx, yy, error_grid, levels=20, cmap='hot')
        axes[2].set_title('Absolute Error', fontsize=12)
        axes[2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[2])
    else:
        # For square
        domain_bounds = example.get_domain_bounds()
        x_min, x_max = domain_bounds[0]
        y_min, y_max = domain_bounds[1]
        
        n_grid = 50
        x_grid = np.linspace(x_min, x_max, n_grid)
        y_grid = np.linspace(y_min, y_max, n_grid)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        grid_points = torch.tensor(
            np.column_stack([xx.flatten(), yy.flatten()]),
            dtype=torch.float32
        ).to(device)
        
        with torch.no_grad():
            u_grid_pred = solver.model(grid_points).cpu().numpy().reshape(xx.shape)
        
        im0 = axes[0].contourf(xx, yy, u_grid_pred, levels=20, cmap='viridis')
        axes[0].set_title('Predicted Solution', fontsize=12)
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0])
        
        u_grid_exact = example.u_exact(xx, yy)
        
        im1 = axes[1].contourf(xx, yy, u_grid_exact, levels=20, cmap='viridis')
        axes[1].set_title('Exact Solution', fontsize=12)
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1])
        
        error_grid = np.abs(u_grid_pred - u_grid_exact)
        
        im2 = axes[2].contourf(xx, yy, error_grid, levels=20, cmap='hot')
        axes[2].set_title('Absolute Error', fontsize=12)
        axes[2].set_aspect('equal')
        plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    arch = 'resnet' if result['use_resnet'] else 'no_resnet'
    plt.savefig(f'results/figures/solution_{result["example"]}_{arch}.png', dpi=150)
    print(f"Saved: results/figures/solution_{result['example']}_{arch}.png")
    plt.close()


if __name__ == '__main__':
    # Run main experiment with square domain
    print("\n" + "="*70)
    print("DEEP RITZ METHOD FOR POISSON EQUATION")
    print("="*70)
    
    # Experiment 1: ResNet with random sampling (main)
    solver1, example1, test_pts1, result1 = run_experiment(
        example_name='square',
        use_resnet=True,
        use_random_sampling=True,
        num_epochs=3000,
        batch_size_interior=256,
        batch_size_boundary=64,
        hidden_dim=64,
        num_blocks=5
    )
    
    plot_results(solver1, example1, test_pts1, result1)
    
    # Experiment 2: Without ResNet (for comparison)
    print("\n" + "="*70)
    print("COMPARISON: Without ResNet")
    print("="*70)
    
    solver2, example2, test_pts2, result2 = run_experiment(
        example_name='square',
        use_resnet=False,
        use_random_sampling=True,
        num_epochs=3000,
        batch_size_interior=256,
        batch_size_boundary=64,
        hidden_dim=64,
        num_blocks=5
    )
    
    plot_results(solver2, example2, test_pts2, result2)
    
    # Print error comparison table
    print("\n" + "="*70)
    print("ERROR COMPARISON")
    print("="*70)
    print(f"{'Method':<30} {'L2 Error':<15} {'L∞ Error':<15}")
    print("-" * 60)
    print(f"{'Deep Ritz (ResNet)':<30} {result1['l2_error']:<15.6e} {result1['linf_error']:<15.6e}")
    print(f"{'Deep Ritz (No ResNet)':<30} {result2['l2_error']:<15.6e} {result2['linf_error']:<15.6e}")
    print("-" * 60)
    
    # Save error results
    os.makedirs('results/errors', exist_ok=True)
    with open('results/errors/error_comparison.txt', 'w') as f:
        f.write("ERROR COMPARISON TABLE\n")
        f.write("="*60 + "\n")
        f.write(f"{'Method':<30} {'L2 Error':<15} {'L∞ Error':<15}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Deep Ritz (ResNet)':<30} {result1['l2_error']:<15.6e} {result1['linf_error']:<15.6e}\n")
        f.write(f"{'Deep Ritz (No ResNet)':<30} {result2['l2_error']:<15.6e} {result2['linf_error']:<15.6e}\n")
        f.write("-"*60 + "\n")
    
    print("\nResults saved to results/ directory")
