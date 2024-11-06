import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.exp((-x**2) / 2)

def f1_derivative(x):
    return -x * np.exp((-x**2) / 2)

def f2(x):
    return np.sin((3 * x**4) / 5)**3

def f2_derivative(x):
    return 12 * x**3 * np.sin((3 * x**4) / 5)**2 * np.cos((3 * x**4) / 5) / 5

def f3(x):
    return np.cos(x / (x + 1))**2

def f3_derivative(x):
    return -2 * np.cos(x / (x + 1)) * np.sin(x / (x + 1)) * (1 / (x + 1)**2)

def f4(x):
    return np.log(x + np.sqrt(4 + x**2))

def f4_derivative(x):
    return (1 + x / np.sqrt(4 + x**2)) / (x + np.sqrt(4 + x**2))

def f5(x):
    return (x * np.arctan(2 * x)) / (x ** 2 + 4)

def f5_derivative(x):
    numerator = np.arctan(2 * x) + (2 * x) / (1 + (2 * x) ** 2)
    denominator = x ** 2 + 4
    return (numerator * denominator - x * np.arctan(2 * x) * 2 * x) / (denominator ** 2)


functions = [f1, f2, f3, f4, f5]
derivatives = [f1_derivative, f2_derivative, f3_derivative, f4_derivative, f5_derivative]
function_names = ['exp(-x^2 / 2)', 'sin^3((3x^4) / 5)', 'cos^2(x / (x + 1))', 'ln(x + sqrt(4 + x^2))',
                  'x * arctan(2x) / (x^2 + 4)']

intervals = [(0, 1), (2, 15), (-5, 5)]
steps = [0.01, 0.005]

for i, (func, derivative) in enumerate(zip(functions, derivatives)):
    for interval in intervals:
        for h in steps:
            x = np.arange(interval[0], interval[1], h)
            y = func(x)

            y_prime_numeric = (y[1:] - y[:-1]) / h
            x_prime = x[:-1]

            y_prime_analytic = derivative(x)

            plt.figure(figsize=(12, 6))
            plt.plot(x_prime, y_prime_numeric, label=f'Численная производная (h = {h})', linestyle='--')
            plt.plot(x, y_prime_analytic, label='Аналитическая производная', linestyle='-')
            plt.title(f'Функция: {function_names[i]} на отрезке {interval}')
            plt.xlabel('x')
            plt.ylabel("y'")
            plt.legend()
            plt.grid(True)
            plt.show()
