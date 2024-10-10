import numpy as np
from scipy.optimize import minimize
import math
from fractions import Fraction

# Adjustable parameters
PARAMS = {
    'optimization_method': 'Nelder-Mead',  # Optimization algorithm
    'max_iterations': 10000,  # Maximum number of iterations for optimization
    'tolerance': 1e-12,  # Tolerance for optimization
    'angle_precision': 6,  # Number of decimal places for angle output
    'fraction_limit': 100,  # Maximum denominator for fraction approximation
    'initial_guess_range': (-2*np.pi, 2*np.pi),  # Range for initial guesses
    'num_restarts': 10,  # Number of random restarts
}

def explain_parameters():
    print("Adjustable Parameters:")
    print("1. optimization_method: The algorithm used for optimization. Options include 'Nelder-Mead', 'BFGS', 'Powell', etc.")
    print("2. max_iterations: Maximum number of iterations for the optimization algorithm.")
    print("3. tolerance: Tolerance for termination of the optimization.")
    print("4. angle_precision: Number of decimal places to display for angle results.")
    print("5. fraction_limit: Maximum denominator when approximating angles as fractions of pi.")
    print("6. initial_guess_range: Range for generating initial random guesses.")
    print("7. num_restarts: Number of times to restart the optimization with different initial guesses.")

def matrix_function(params):
    phi1, phi2, phi3, phiA, phiB, phiC, phiX, phiY, phiZ = params
    return np.array([
        [np.exp(1j * (phi2 + phiA + phiX)) / np.sqrt(3),
         1j * np.exp(1j * (phi2 + phiB + phiX)) / np.sqrt(3),
         1j * np.exp(1j * (phiC + phiX)) / np.sqrt(3)],
        [(1j/6) * np.exp(1j * (phiA + phiY)) * (3*np.exp(1j*phi1) + 1j*np.sqrt(3)*np.exp(1j*(phi2+phi3))),
         (1/6) * np.exp(1j * (phiB + phiY)) * (3*np.exp(1j*phi1) - 1j*np.sqrt(3)*np.exp(1j*(phi2+phi3))),
         1j * np.exp(1j * (phi3 + phiC + phiY)) / np.sqrt(3)],
        [(1/6) * np.exp(1j * (phiA + phiZ)) * (1j*np.sqrt(3)*np.exp(1j*(phi2+phi3)) - 3*np.exp(1j*phi1)),
         (1j/6) * np.exp(1j * (phiB + phiZ)) * (3*np.exp(1j*phi1) + 1j*np.sqrt(3)*np.exp(1j*(phi2+phi3))),
         np.exp(1j * (phi3 + phiC + phiZ)) / np.sqrt(3)]
    ])

target_matrix = (1/np.sqrt(3)) * np.array([
    [1, 1, 1],
    [1, np.exp(2j*np.pi/3), np.exp(4j*np.pi/3)],
    [1, np.exp(4j*np.pi/3), np.exp(2j*np.pi/3)]
])

def matrix_difference(params):
    return np.abs(matrix_function(params) - target_matrix).sum()

def interpret_angle(angle):
    angle = angle % (2*np.pi)
    radians = round(angle, PARAMS['angle_precision'])
    degrees = round(np.degrees(angle), PARAMS['angle_precision'])
    fraction = Fraction(angle / np.pi).limit_denominator(PARAMS['fraction_limit'])
    return f"{radians} rad, {degrees}°, {fraction}π"

def optimize_matrix():
    best_result = None
    best_error = float('inf')

    for i in range(PARAMS['num_restarts']):
        initial_guess = np.random.uniform(*PARAMS['initial_guess_range'], 9)

        result = minimize(matrix_difference, initial_guess,
                          method=PARAMS['optimization_method'],
                          options={'maxiter': PARAMS['max_iterations'],
                                   'xatol': PARAMS['tolerance'],
                                   'fatol': PARAMS['tolerance']})

        if result.fun < best_error:
            best_result = result
            best_error = result.fun

        print(f"Restart {i+1}/{PARAMS['num_restarts']}: Current best error = {best_error}")

    return best_result

def main():
    explain_parameters()
    print("\nStarting optimization...")
    result = optimize_matrix()

    print("\nOptimization Results:")
    print(f"Success: {result.success}")
    print(f"Final error: {result.fun}")

    variable_names = ['phi1', 'phi2', 'phi3', 'phiA', 'phiB', 'phiC', 'phiX', 'phiY', 'phiZ']
    print("\nOptimized angles:")
    for name, value in zip(variable_names, result.x):
        print(f"{name}: {interpret_angle(value)}")

    print("\nResulting matrix M1:")
    print(matrix_function(result.x))

    print("\nTarget matrix M2:")
    print(target_matrix)

if __name__ == "__main__":
    main()