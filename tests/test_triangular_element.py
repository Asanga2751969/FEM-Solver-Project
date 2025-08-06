"""
Test script for TriangularElement class
Tests basic functionality and validates against analytical solutions
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to path so we can import fem_core
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fem_core.elements.triangular import TriangularElement

def test_basic_functionality():
    """Test basic element creation and properties"""
    print("=== Test 1: Basic Functionality ===")
    
    # Create a right triangle
    nodes = np.array([
        [0.0, 0.0],  # Node 0
        [2.0, 0.0],  # Node 1  
        [0.0, 1.0]   # Node 2
    ])
    
    material = {'E': 200e9, 'nu': 0.3}  # Steel properties
    
    try:
        element = TriangularElement(nodes, material, element_id=1)
        print("✅ Element created successfully")
        
        # Check area calculation
        expected_area = 0.5 * 2.0 * 1.0  # 0.5 * base * height
        assert abs(element.area - expected_area) < 1e-10, f"Area mismatch: {element.area} vs {expected_area}"
        print(f"✅ Area calculation correct: {element.area}")
        
        # Check matrix dimensions
        assert element.K_element.shape == (6, 6), f"Stiffness matrix wrong shape: {element.K_element.shape}"
        assert element.B_matrix.shape == (3, 6), f"B matrix wrong shape: {element.B_matrix.shape}"
        assert element.D_matrix.shape == (3, 3), f"D matrix wrong shape: {element.D_matrix.shape}"
        print("✅ Matrix dimensions correct")
        
        # Check symmetry of stiffness matrix
        K_diff = element.K_element - element.K_element.T
        assert np.allclose(K_diff, np.zeros_like(K_diff), atol=1e-12), "Stiffness matrix not symmetric"
        print("✅ Stiffness matrix is symmetric")
        
        print("Test 1 PASSED ✅\n")
        return True
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}\n")
        return False

def test_rigid_body_motion():
    """Test that rigid body motions produce no stresses"""
    print("=== Test 2: Rigid Body Motion ===")
    
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0], 
        [0.5, 0.866]  # Equilateral triangle
    ])
    
    material = {'E': 200e9, 'nu': 0.3}
    element = TriangularElement(nodes, material)
    
    try:
        # Test 1: Pure translation
        translation = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Same displacement for all nodes
        stresses_translation = element.get_element_stresses(translation)
        assert np.allclose(stresses_translation, np.zeros(3), atol=1e-10), \
            f"Translation produced stresses: {stresses_translation}"
        print("✅ Pure translation produces zero stress")
        
        # Test 2: Pure rotation (small angle approximation)
        theta = 0.01  # Small rotation angle in radians
        rotation = np.array([
            -nodes[0,1]*theta, nodes[0,0]*theta,  # Node 0
            -nodes[1,1]*theta, nodes[1,0]*theta,  # Node 1  
            -nodes[2,1]*theta, nodes[2,0]*theta   # Node 2
        ])
        stresses_rotation = element.get_element_stresses(rotation)
        assert np.allclose(stresses_rotation, np.zeros(3), atol=1e-8), \
            f"Rotation produced stresses: {stresses_rotation}"
        print("✅ Pure rotation produces zero stress")
        
        print("Test 2 PASSED ✅\n")
        return True
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}\n")
        return False

def test_uniaxial_tension():
    """Test uniaxial tension against analytical solution"""
    print("=== Test 3: Uniaxial Tension Test ===")
    
    # Create a rectangular element (using triangle)
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    E = 200e9  # Young's modulus
    nu = 0.3   # Poisson's ratio
    material = {'E': E, 'nu': nu}
    element = TriangularElement(nodes, material)
    
    try:
        # Apply uniform strain in x-direction: εx = 0.001, εy = 0, γxy = 0
        strain_x = 0.001
        
        # For this strain state, the displacements should be:
        # u = εx * x, v = -ν * εx * y (from Poisson effect)
        displacements = np.array([
            strain_x * nodes[0,0], -nu * strain_x * nodes[0,1],  # Node 0
            strain_x * nodes[1,0], -nu * strain_x * nodes[1,1],  # Node 1
            strain_x * nodes[2,0], -nu * strain_x * nodes[2,1]   # Node 2
        ])
        
        stresses = element.get_element_stresses(displacements)
        strains = element.get_element_strains(displacements)
        
        # Expected results for plane stress
        expected_stress_x = E * strain_x  # σx = E * εx
        expected_stress_y = 0.0           # σy = 0 (plane stress)
        expected_shear = 0.0              # τxy = 0
        
        expected_strain_x = strain_x
        expected_strain_y = -nu * strain_x  # εy = -ν * εx (Poisson effect)
        expected_shear_strain = 0.0
        
        print(f"Calculated stresses: σx={stresses[0]:.2e}, σy={stresses[1]:.2e}, τxy={stresses[2]:.2e}")
        print(f"Expected stresses:   σx={expected_stress_x:.2e}, σy={expected_stress_y:.2e}, τxy={expected_shear:.2e}")
        
        # Check stresses (with some tolerance for numerical errors)
        assert abs(stresses[0] - expected_stress_x) / expected_stress_x < 1e-10, \
            f"σx mismatch: {stresses[0]} vs {expected_stress_x}"
        assert abs(stresses[1] - expected_stress_y) < 1e6, \
            f"σy should be zero: {stresses[1]}"  # Small tolerance for numerical errors
        assert abs(stresses[2] - expected_shear) < 1e6, \
            f"τxy should be zero: {stresses[2]}"
        
        print("✅ Uniaxial tension stresses correct")
        print("Test 3 PASSED ✅\n")
        return True
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}\n")
        return False

def test_patch_test():
    """Patch test - constant strain should be reproduced exactly"""
    print("=== Test 4: Patch Test ===")
    
    # Create element with arbitrary shape
    nodes = np.array([
        [0.2, 0.1],
        [1.3, 0.3],
        [0.8, 1.2]
    ])
    
    material = {'E': 200e9, 'nu': 0.3}
    element = TriangularElement(nodes, material)
    
    try:
        # Define a constant strain state
        target_strains = np.array([0.002, -0.001, 0.0005])  # [εx, εy, γxy]
        
        # Calculate displacements that would produce this strain state
        # For linear elements, if we can find ANY displacement field that gives
        # constant strain, the element should reproduce it exactly
        
        # Use a simple linear displacement field
        # u = εx*x + 0.5*γxy*y
        # v = εy*y + 0.5*γxy*x  
        εx, εy, γxy = target_strains
        
        displacements = np.zeros(6)
        for i in range(3):
            x, y = nodes[i]
            displacements[2*i] = εx * x + 0.5 * γxy * y      # u
            displacements[2*i+1] = εy * y + 0.5 * γxy * x    # v
        
        calculated_strains = element.get_element_strains(displacements)
        
        print(f"Target strains:     {target_strains}")
        print(f"Calculated strains: {calculated_strains}")
        
        strain_error = np.abs(calculated_strains - target_strains)
        assert np.all(strain_error < 1e-12), f"Strain reproduction error: {strain_error}"
        
        print("✅ Constant strain reproduced exactly")
        print("Test 4 PASSED ✅\n")
        return True
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}\n")
        return False

def visualize_test_element():
    """Create a visualization of test elements"""
    print("=== Visualization Test ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Test different element shapes
    test_cases = [
        {
            'name': 'Right Triangle',
            'nodes': np.array([[0, 0], [2, 0], [0, 1]]),
            'ax': axes[0]
        },
        {
            'name': 'Equilateral Triangle',
            'nodes': np.array([[0, 0], [1, 0], [0.5, 0.866]]),
            'ax': axes[1]
        },
        {
            'name': 'Obtuse Triangle', 
            'nodes': np.array([[0, 0], [2, 0], [0.3, 0.8]]),
            'ax': axes[2]
        },
        {
            'name': 'Irregular Triangle',
            'nodes': np.array([[0.2, 0.1], [1.8, 0.3], [0.7, 1.4]]),
            'ax': axes[3]
        }
    ]
    
    material = {'E': 200e9, 'nu': 0.3}
    
    for case in test_cases:
        try:
            element = TriangularElement(case['nodes'], material)
            element.plot_element(ax=case['ax'], show_nodes=True, show_element_id=False)
            case['ax'].set_title(f"{case['name']}\nArea: {element.area:.3f}")
        except Exception as e:
            case['ax'].text(0.5, 0.5, f"ERROR:\n{str(e)}", 
                          transform=case['ax'].transAxes, ha='center', va='center')
            case['ax'].set_title(f"{case['name']} - FAILED")
    
    plt.tight_layout()
    plt.savefig('triangular_element_tests.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization complete - check 'triangular_element_tests.png'")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Triangular Element Tests\n" + "="*50)
    
    tests = [
        test_basic_functionality,
        test_rigid_body_motion, 
        test_uniaxial_tension,
        test_patch_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Your triangular element is working correctly.")
    else:
        print("⚠️  Some tests FAILED. Check the implementation.")
    
    # Always run visualization
    print("\n" + "="*50)
    visualize_test_element()
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
