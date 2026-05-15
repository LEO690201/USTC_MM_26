"""
Deep Ritz solver for Poisson equation.
Implements the loss function and training procedure.
"""

import torch
import torch.nn as nn
import numpy as np
from network import DeepRitzNetwork, DeepRitzNetworkNoResNet


class DeepRitzSolver:
    """Solver for Poisson equation using Deep Ritz method."""
    
    def __init__(self, model, domain_bounds, boundary_points_func, 
                 f_func, g_func, penalty_beta=100.0, device='cpu'):
        """
        Args:
            model: Neural network model
            domain_bounds: Tuple ((x_min, x_max), (y_min, y_max)) for 2D
            boundary_points_func: Function to generate boundary points
            f_func: Source function f(x, y)
            g_func: Boundary condition function g(x, y)
            penalty_beta: Penalty parameter for boundary condition
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.domain_bounds = domain_bounds
        self.boundary_points_func = boundary_points_func
        self.f_func = f_func
        self.g_func = g_func
        self.penalty_beta = penalty_beta
        
        self.loss_history = []
    
    def compute_gradients(self, x, compute_hessian=False):
        """
        Compute first and optionally second derivatives of u w.r.t. x.
        
        Args:
            x: Input points, shape (batch_size, 2)
            compute_hessian: If True, also compute Hessian (∇²u)
            
        Returns:
            u: Network output, shape (batch_size, 1)
            grad_u: First derivatives, shape (batch_size, 2)
            hessian: Hessian matrix if compute_hessian=True, else None
        """
        x.requires_grad_(True)
        
        u = self.model(x)
        
        # Compute ∇u (first derivatives)
        grad_u = torch.autograd.grad(
            outputs=u.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        if compute_hessian:
            # Compute ∇²u (Laplacian)
            # For 2D: ∇²u = ∂²u/∂x² + ∂²u/∂y²
            u_xx = torch.autograd.grad(
                outputs=grad_u[:, 0].sum(),
                inputs=x,
                create_graph=False,
                retain_graph=True
            )[0][:, 0]
            
            u_yy = torch.autograd.grad(
                outputs=grad_u[:, 1].sum(),
                inputs=x,
                create_graph=False,
                retain_graph=True
            )[0][:, 1]
            
            laplacian = u_xx + u_yy
            return u, grad_u, laplacian
        
        return u, grad_u, None
    
    def compute_loss(self, interior_points, boundary_points, use_hessian=True):
        """
        Compute the loss function:
        L(θ) = ∫(1/2|∇u|² - f·u) dx + β∫(u - g)² dS
        
        Using numerical quadrature (Monte Carlo or Gauss-Legendre).
        
        Args:
            interior_points: Points inside domain, shape (N_interior, 2)
            boundary_points: Points on boundary, shape (N_boundary, 2)
            use_hessian: If True, use Laplacian; if False, use |∇u|²
            
        Returns:
            loss: Scalar loss value
            interior_loss: Loss from interior
            boundary_loss: Loss from boundary
        """
        # Interior points
        interior_points = interior_points.to(self.device)
        boundary_points = boundary_points.to(self.device)
        
        # Compute on interior
        if use_hessian:
            u_int, grad_u_int, laplacian = self.compute_gradients(
                interior_points, compute_hessian=True
            )
            
            f_val = torch.tensor([
                self.f_func(interior_points[i, 0].item(), interior_points[i, 1].item())
                for i in range(interior_points.shape[0])
            ]).to(self.device).reshape(-1, 1)
            
            # Interior loss using Laplacian
            # L_interior = ∫(1/2|∇u|² - f·u) dx
            grad_norm_sq = (grad_u_int ** 2).sum(dim=1, keepdim=True)
            interior_loss = 0.5 * grad_norm_sq - f_val * u_int
            interior_loss = interior_loss.mean() * (
                (self.domain_bounds[0][1] - self.domain_bounds[0][0]) *
                (self.domain_bounds[1][1] - self.domain_bounds[1][0])
            )
        else:
            u_int, grad_u_int, _ = self.compute_gradients(
                interior_points, compute_hessian=False
            )
            
            f_val = torch.tensor([
                self.f_func(interior_points[i, 0].item(), interior_points[i, 1].item())
                for i in range(interior_points.shape[0])
            ]).to(self.device).reshape(-1, 1)
            
            grad_norm_sq = (grad_u_int ** 2).sum(dim=1, keepdim=True)
            interior_loss = 0.5 * grad_norm_sq - f_val * u_int
            interior_loss = interior_loss.mean() * (
                (self.domain_bounds[0][1] - self.domain_bounds[0][0]) *
                (self.domain_bounds[1][1] - self.domain_bounds[1][0])
            )
        
        # Compute on boundary
        u_bound = self.model(boundary_points)
        g_val = torch.tensor([
            self.g_func(boundary_points[i, 0].item(), boundary_points[i, 1].item())
            for i in range(boundary_points.shape[0])
        ]).to(self.device).reshape(-1, 1)
        
        boundary_loss = self.penalty_beta * ((u_bound - g_val) ** 2).mean()
        
        total_loss = interior_loss + boundary_loss
        
        return total_loss, interior_loss, boundary_loss
    
    def train(self, num_epochs, batch_size_interior, batch_size_boundary,
              learning_rate=0.001, verbose=True):
        """
        Training loop using SGD with random sampling.
        
        Args:
            num_epochs: Number of training epochs
            batch_size_interior: Number of interior points per batch
            batch_size_boundary: Number of boundary points per batch
            learning_rate: Learning rate for optimizer
            verbose: Print loss during training
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            # Sample random interior points
            interior_points = self.sample_interior_points(batch_size_interior)
            
            # Sample random boundary points
            boundary_points = self.sample_boundary_points(batch_size_boundary)
            
            # Compute loss
            loss, int_loss, bound_loss = self.compute_loss(
                interior_points, boundary_points, use_hessian=True
            )
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if verbose and (epoch + 1) % max(1, num_epochs // 20) == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {loss.item():.6e}, "
                      f"Interior: {int_loss.item():.6e}, "
                      f"Boundary: {bound_loss.item():.6e}")
    
    def sample_interior_points(self, num_points):
        """Sample random points uniformly from interior domain."""
        x_min, x_max = self.domain_bounds[0]
        y_min, y_max = self.domain_bounds[1]
        
        x = torch.rand(num_points, 1) * (x_max - x_min) + x_min
        y = torch.rand(num_points, 1) * (y_max - y_min) + y_min
        
        return torch.cat([x, y], dim=1)
    
    def sample_boundary_points(self, num_points):
        """Sample random points from boundary."""
        return self.boundary_points_func(num_points)
    
    def predict(self, x):
        """
        Predict u at given points.
        
        Args:
            x: Input points, shape (batch_size, 2)
            
        Returns:
            u: Predictions, shape (batch_size, 1)
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x)
    
    def compute_l2_error(self, test_points, exact_solution_func):
        """
        Compute L2 error between predicted and exact solution.
        
        Args:
            test_points: Test points, shape (N, 2)
            exact_solution_func: Function returning exact solution
            
        Returns:
            l2_error: L2 norm of error
        """
        u_pred = self.predict(test_points)
        
        u_exact = torch.tensor([
            exact_solution_func(test_points[i, 0].item(), test_points[i, 1].item())
            for i in range(test_points.shape[0])
        ]).reshape(-1, 1).to(self.device)
        
        error = u_pred - u_exact
        l2_error = torch.sqrt((error ** 2).mean()).item()
        
        return l2_error
    
    def compute_linf_error(self, test_points, exact_solution_func):
        """
        Compute L∞ error between predicted and exact solution.
        
        Args:
            test_points: Test points, shape (N, 2)
            exact_solution_func: Function returning exact solution
            
        Returns:
            linf_error: L∞ norm of error
        """
        u_pred = self.predict(test_points)
        
        u_exact = torch.tensor([
            exact_solution_func(test_points[i, 0].item(), test_points[i, 1].item())
            for i in range(test_points.shape[0])
        ]).reshape(-1, 1).to(self.device)
        
        error = torch.abs(u_pred - u_exact)
        linf_error = error.max().item()
        
        return linf_error
