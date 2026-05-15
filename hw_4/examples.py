"""
Poisson equation examples with analytical solutions.
"""

import numpy as np
import torch


class PoissonExample:
    """Base class for Poisson equation examples."""
    
    def __init__(self, name):
        self.name = name
    
    def f(self, x, y):
        """Source function: f(x, y)"""
        raise NotImplementedError
    
    def g(self, x, y):
        """Boundary condition: u(x, y) = g(x, y) on ∂Ω"""
        raise NotImplementedError
    
    def u_exact(self, x, y):
        """Exact solution: u(x, y)"""
        raise NotImplementedError
    
    def get_domain_bounds(self):
        """Returns ((x_min, x_max), (y_min, y_max))"""
        raise NotImplementedError
    
    def boundary_points(self, num_points):
        """Generate boundary points."""
        raise NotImplementedError


class SquarePoissonExample(PoissonExample):
    """
    Poisson equation on unit square [0,1]²
    -∇²u = f  in Ω
    u = g     on ∂Ω
    
    Example: -∇²u = -2π²sin(πx)sin(πy)
    with exact solution u(x,y) = sin(πx)sin(πy)
    """
    
    def __init__(self):
        super().__init__("Unit Square Poisson")
    
    def f(self, x, y):
        """Source: -∇²u where u = sin(πx)sin(πy)"""
        return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def g(self, x, y):
        """Boundary condition: u = 0 on all boundaries"""
        return 0.0
    
    def u_exact(self, x, y):
        """Exact solution: sin(πx)sin(πy)"""
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def get_domain_bounds(self):
        return ((0.0, 1.0), (0.0, 1.0))
    
    def boundary_points(self, num_points):
        """Generate points on boundary of [0,1]²"""
        points = []
        
        # Points per boundary (equally distributed)
        n_per_side = num_points // 4
        
        # Bottom boundary (y=0)
        x = np.linspace(0, 1, n_per_side, endpoint=False)
        points.extend([(xi, 0.0) for xi in x])
        
        # Top boundary (y=1)
        points.extend([(xi, 1.0) for xi in x])
        
        # Left boundary (x=0)
        y = np.linspace(0, 1, n_per_side, endpoint=False)
        points.extend([(0.0, yi) for yi in y])
        
        # Right boundary (x=1)
        points.extend([(1.0, yi) for yi in y])
        
        # Handle remaining points
        remaining = num_points - len(points)
        if remaining > 0:
            extra_x = np.linspace(0, 1, remaining, endpoint=False)
            points.extend([(xi, 0.0) for xi in extra_x])
        
        # Convert to tensor
        points = np.array(points)
        # Add small random noise to avoid boundary singularities
        noise = np.random.randn(*points.shape) * 1e-6
        points = np.clip(points + noise, [0, 0], [1, 1])
        
        return torch.tensor(points, dtype=torch.float32)


class SquareHomogeneousPoissonExample(PoissonExample):
    """
    Poisson equation on unit square with non-homogeneous boundary.
    -∇²u = f  in Ω
    u = g     on ∂Ω
    
    Example with non-zero boundary condition.
    Exact solution: u(x,y) = x(1-x)y(1-y) + x
    Source: f(x,y) = -∇²u
    """
    
    def __init__(self):
        super().__init__("Square Non-homogeneous Poisson")
    
    def u_exact(self, x, y):
        """Exact solution"""
        return x * (1 - x) * y * (1 - y) + x
    
    def f(self, x, y):
        """Source function: f = -∇²u"""
        # ∇²u = ∂²u/∂x² + ∂²u/∂y²
        # u = xy(1-x)(1-y) + x
        # ∂u/∂x = y(1-x)(1-y) - xy(1-y) + 1 = y(1-2x)(1-y) + 1
        # ∂²u/∂x² = -2y(1-y)
        # ∂u/∂y = x(1-x)(1-y) - x(1-x)y = x(1-x)(1-2y)
        # ∂²u/∂y² = -2x(1-x)
        # ∇²u = -2y(1-y) - 2x(1-x)
        laplacian = -2 * y * (1 - y) - 2 * x * (1 - x)
        return -laplacian
    
    def g(self, x, y):
        """Boundary condition"""
        return self.u_exact(x, y)
    
    def get_domain_bounds(self):
        return ((0.0, 1.0), (0.0, 1.0))
    
    def boundary_points(self, num_points):
        """Generate boundary points with correct values"""
        points = []
        
        n_per_side = num_points // 4
        
        # Bottom boundary (y=0)
        x = np.linspace(0, 1, n_per_side, endpoint=False)
        points.extend([(xi, 0.0) for xi in x])
        
        # Top boundary (y=1)
        points.extend([(xi, 1.0) for xi in x])
        
        # Left boundary (x=0)
        y = np.linspace(0, 1, n_per_side, endpoint=False)
        points.extend([(0.0, yi) for yi in y])
        
        # Right boundary (x=1)
        points.extend([(1.0, yi) for yi in y])
        
        remaining = num_points - len(points)
        if remaining > 0:
            extra_x = np.linspace(0, 1, remaining, endpoint=False)
            points.extend([(xi, 0.0) for xi in extra_x])
        
        points = np.array(points)
        noise = np.random.randn(*points.shape) * 1e-6
        points = np.clip(points + noise, [0, 0], [1, 1])
        
        return torch.tensor(points, dtype=torch.float32)


class CircularDiskPoissonExample(PoissonExample):
    """
    Poisson equation on circular disk.
    Domain: Ω = {(x,y): x² + y² ≤ 1}
    """
    
    def __init__(self):
        super().__init__("Circular Disk Poisson")
    
    def u_exact(self, x, y):
        """Exact solution: u = 1 - (x² + y²)"""
        return 1 - (x**2 + y**2)
    
    def f(self, x, y):
        """Source: f = -∇²u = 4 for u = 1 - r²"""
        return 4.0
    
    def g(self, x, y):
        """Boundary condition: u = 0 on boundary"""
        return 0.0
    
    def get_domain_bounds(self):
        return ((-1.0, 1.0), (-1.0, 1.0))
    
    def boundary_points(self, num_points):
        """Generate points on circle boundary"""
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        
        points = np.column_stack([x, y])
        
        # Add small inward perturbation to avoid exact boundary issues
        noise = np.random.randn(*points.shape) * 1e-6
        r = np.sqrt((points[:, 0])**2 + (points[:, 1])**2)
        r = np.clip(r + noise[:, 0] * 0.01, 0.99, 1.0)
        points = points / r[:, None] * r[:, None]
        
        return torch.tensor(points, dtype=torch.float32)


def create_example(example_type='square'):
    """Factory function to create Poisson examples."""
    if example_type == 'square':
        return SquarePoissonExample()
    elif example_type == 'square_inhom':
        return SquareHomogeneousPoissonExample()
    elif example_type == 'circle':
        return CircularDiskPoissonExample()
    else:
        raise ValueError(f"Unknown example type: {example_type}")
