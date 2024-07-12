import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the service time distribution
mean_service_time = 10  # seconds
scv = 16  # Squared coefficient of variation

# Calculate variance based on mean service time and SCV
variance_service_time = scv * mean_service_time ** 2

# Calculate Pareto parameters
def calculate_pareto_parameters(mean, variance):
    alpha = (2 * mean ** 2 + variance) / (variance - mean ** 2)
    xm = (alpha - 1) * mean / alpha
    return xm, alpha

xm, alpha = calculate_pareto_parameters(mean_service_time, variance_service_time)

# Generate service times from Pareto distribution
def generate_service_times(n, xm, alpha):
    return (np.random.pareto(alpha, n) + 1) * xm

# Simulate the M/G/1/FCFS queue
def simulate_queue(lambda_rate, num_jobs=1000000, num_runs=10):
    mean_response_times = []
    
    for _ in range(num_runs):
        inter_arrival_times = np.random.exponential(scale=1/lambda_rate, size=num_jobs)
        arrival_times = np.cumsum(inter_arrival_times)
        service_times = generate_service_times(num_jobs, xm, alpha)
        
        start_times = np.zeros(num_jobs)
        finish_times = np.zeros(num_jobs)
        response_times = np.zeros(num_jobs)
        
        for i in range(1, num_jobs):
            start_times[i] = max(arrival_times[i], finish_times[i-1])
            finish_times[i] = start_times[i] + service_times[i]
            response_times[i] = finish_times[i] - arrival_times[i]
        
        mean_response_times.append(np.mean(response_times))
    
    return np.mean(mean_response_times), np.std(mean_response_times)

# Theoretical mean response time
def theoretical_mean_response(lambda_rate):
    rho = lambda_rate * mean_service_time
    E_T = mean_service_time * (1 + (lambda_rate * variance_service_time) / (2 * (1 - rho)))
    return E_T

# Define range of lambda values
lambda_values = np.arange(0.01, 0.1, 0.01)
simulated_mean_responses = []
simulated_std_responses = []
theoretical_mean_responses = []

# Run simulations
for lambda_rate in lambda_values:
    sim_mean, sim_std = simulate_queue(lambda_rate)
    simulated_mean_responses.append(sim_mean)
    simulated_std_responses.append(sim_std)
    theoretical_mean_responses.append(theoretical_mean_response(lambda_rate))

# Plot results
plt.figure(figsize=(12, 6))
plt.errorbar(lambda_values, simulated_mean_responses, yerr=simulated_std_responses, label='Simulated Mean Response Time', marker='o', linestyle='none', capsize=5)
plt.plot(lambda_values, theoretical_mean_responses, label='Theoretical Mean Response Time', marker='x')
plt.xlabel('Arrival Rate (lambda)')
plt.ylabel('Mean Response Time (E[T])')
plt.title('M/G/1/FCFS Queue Mean Response Time with Pareto Distribution')
plt.legend()
plt.grid(True)
plt.show()
