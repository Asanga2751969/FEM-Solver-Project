import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class TriangularElement:
    """
    3-node triangular element for 2D linear elasticity (plane stress/strain)
    
    Node numbering convention:
        2
        |\\
        | \\
        |  \\
        |   \\
        0----1
    """
    
    def __init__(self, nodes: np.ndarray, material_props: dict, element_id: int = 0):
        """
        Initialize triangular element
        
        Parameters:
        -----------
        nodes : np.ndarray, shape (3, 2)
            Coordinates of the 3 nodes [[x0, y0], [x1, y1], [x2, y2]]
        material_props : dict
            Material properties with keys 'E' (Young's modulus), 'nu' (Poisson's ratio)
        element_id : int
            Element identifier
        """
        self.nodes = np.array(nodes)
        self.E = material_props['E']  # Young's modulus
        self.nu = material_props['nu']  # Poisson's ratio
        self.element_id = element_id
        
        # Validate input
        if self.nodes.shape != (3, 2):
            raise ValueError("nodes must be a 3x2 array")
        
        # Calculate geometric properties
        self.area = self._calculate_area()
        if self.area <= 0:
            raise ValueError("Element has zero or negative area - check node ordering")
        
        # Calculate shape function derivatives (constant for linear triangular elements)
        self.B_matrix = self._calculate_B_matrix()
        
        # Calculate material matrix (plane stress assumed)
        self.D_matrix = self._calculate_material_matrix()
        
        # Calculate element stiffness matrix
        self.K_element = self._calculate_stiffness_matrix()
    
    def _calculate_area(self) -> float:
        """
        Calculate element area using cross product
        
        For triangle with vertices (x0,y0), (x1,y1), (x2,y2):
        Area = 0.5 * |det([[x0, y0, 1], [x1, y1, 1], [x2, y2, 1]])|
        """
        x0, y0 = self.nodes[0]
        x1, y1 = self.nodes[1]
        x2, y2 = self.nodes[2]
        
        # Using the determinant formula
        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        return area
    
    def _calculate_B_matrix(self) -> np.ndarray:
        """
        Calculate strain-displacement matrix B
        
        For linear triangular elements, strains are constant throughout the element:
        {ε} = [B]{u} where {ε} = {εx, εy, γxy}^T and {u} = {u0, v0, u1, v1, u2, v2}^T
        
        B matrix relates nodal displacements to element strains
        """
        x0, y0 = self.nodes[0]
        x1, y1 = self.nodes[1]
        x2, y2 = self.nodes[2]
        
        # Shape function derivatives (these are constants for linear triangles)
        # ∂N0/∂x, ∂N1/∂x, ∂N2/∂x
        dN_dx = np.array([y1 - y2, y2 - y0, y0 - y1]) / (2 * self.area)
        # ∂N0/∂y, ∂N1/∂y, ∂N2/∂y  
        dN_dy = np.array([x2 - x1, x0 - x2, x1 - x0]) / (2 * self.area)
        
        # Construct B matrix [3 x 6]
        # Row 1: εx = ∂u/∂x
        # Row 2: εy = ∂v/∂y  
        # Row 3: γxy = ∂u/∂y + ∂v/∂x
        B = np.zeros((3, 6))
        
        for i in range(3):
            # u-displacement contributions
            B[0, 2*i] = dN_dx[i]      # εx = ∂u/∂x
            B[2, 2*i] = dN_dy[i]      # γxy = ∂u/∂y + ∂v/∂x
            
            # v-displacement contributions  
            B[1, 2*i+1] = dN_dy[i]    # εy = ∂v/∂y
            B[2, 2*i+1] = dN_dx[i]    # γxy = ∂u/∂y + ∂v/∂x
        
        return B
    
    def _calculate_material_matrix(self) -> np.ndarray:
        """
        Calculate material property matrix D for plane stress
        
        For plane stress: σz = 0, but εz ≠ 0
        [D] relates stress to strain: {σ} = [D]{ε}
        """
        factor = self.E / (1 - self.nu**2)
        
        D = factor * np.array([
            [1,      self.nu, 0],
            [self.nu, 1,      0],
            [0,       0,      (1 - self.nu) / 2]
        ])
        
        return D
    
    def _calculate_stiffness_matrix(self) -> np.ndarray:
        """
        Calculate element stiffness matrix
        
        [K] = ∫∫ [B]^T [D] [B] t dA
        
        For constant thickness and linear triangular elements:
        [K] = [B]^T [D] [B] * Area * thickness
        """
        thickness = 1.0  # Assuming unit thickness for now
        K = self.B_matrix.T @ self.D_matrix @ self.B_matrix * self.area * thickness
        return K
    
    def get_element_stresses(self, element_displacements: np.ndarray) -> np.ndarray:
        """
        Calculate element stresses given nodal displacements
        
        Parameters:
        -----------
        element_displacements : np.ndarray, shape (6,)
            Element displacements [u0, v0, u1, v1, u2, v2]
        
        Returns:
        --------
        stresses : np.ndarray, shape (3,)
            Element stresses [σx, σy, τxy]
        """
        strains = self.B_matrix @ element_displacements
        stresses = self.D_matrix @ strains
        return stresses
    
    def get_element_strains(self, element_displacements: np.ndarray) -> np.ndarray:
        """
        Calculate element strains given nodal displacements
        
        Returns:
        --------
        strains : np.ndarray, shape (3,)
            Element strains [εx, εy, γxy]
        """
        strains = self.B_matrix @ element_displacements
        return strains
    
    def plot_element(self, ax=None, show_nodes=True, show_element_id=True):
        """
        Plot the triangular element
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Close the triangle by repeating the first node
        x_coords = np.append(self.nodes[:, 0], self.nodes[0, 0])
        y_coords = np.append(self.nodes[:, 1], self.nodes[0, 1])
        
        ax.plot(x_coords, y_coords, 'b-', linewidth=2)
        ax.fill(x_coords, y_coords, alpha=0.3, color='lightblue')
        
        if show_nodes:
            ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'ro', markersize=8)
            for i, (x, y) in enumerate(self.nodes):
                ax.annotate(f'Node {i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10)
        
        if show_element_id:
            centroid_x = np.mean(self.nodes[:, 0])
            centroid_y = np.mean(self.nodes[:, 1])
            ax.annotate(f'Element {self.element_id}', (centroid_x, centroid_y), 
                       ha='center', va='center', fontweight='bold')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Triangular Element')
        
        return ax
    
    def info(self):
        """Print element information"""
        print(f"=== Triangular Element {self.element_id} ===")
        print(f"Nodes coordinates:")
        for i, node in enumerate(self.nodes):
            print(f"  Node {i}: ({node[0]:.3f}, {node[1]:.3f})")
        print(f"Area: {self.area:.6f}")
        print(f"Material: E = {self.E:.2e}, ν = {self.nu:.3f}")
        print(f"Stiffness matrix shape: {self.K_element.shape}")


# Example usage and testing
if __name__ == "__main__":
    # Define a simple triangular element
    nodes = np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1  
        [0.5, 1.0]   # Node 2
    ])
    
    # Material properties for steel
    material = {
        'E': 200e9,  # 200 GPa
        'nu': 0.3    # Poisson's ratio
    }
    
    # Create element
    element = TriangularElement(nodes, material, element_id=1)
    
    # Print element information
    element.info()
    
    # Plot the element
    element.plot_element()
    plt.show()
    
    # Test with some displacements
    test_displacements = np.array([0.001, 0.0, 0.002, 0.001, 0.0015, 0.0005])
    stresses = element.get_element_stresses(test_displacements)
    strains = element.get_element_strains(test_displacements)
    
    print(f"\nTest results:")
    print(f"Displacements: {test_displacements}")
    print(f"Strains [εx, εy, γxy]: {strains}")
    print(f"Stresses [σx, σy, τxy]: {stresses}")
