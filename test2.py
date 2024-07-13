import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Simulation parameters
E_S = 10  # Mean service time (seconds)
C_S2 = 16  # Squared coefficient of variation
num_jobs = 10000  # Number of jobs to simulate

# Pareto distribution parameters to match the desired mean and variance
alpha = 1 / C_S2
xm = (E_S * (alpha - 1)) / alpha

# Define the arrival rates for utilization 0.5 and 0.9
rho_values = [0.5, 0.9]
lambda_values = [rho / E_S for rho in rho_values]
mean_response_times_sita = []
theoretical_response_times_sita = []
mean_response_times_single = []
theoretical_response_times_single = []

# Service rates for the two servers
mu1 = 0.05
mu2 = 0.05

# Cutoff point for job size
cutoff = 10  # Experiment with different values

# Function to simulate M/G/1 queue
def simulate_mg1(lambda_, mu, num_jobs):
    interarrival_times = np.random.exponential(1 / lambda_, num_jobs)
    arrival_times = np.cumsum(interarrival_times)
    service_times = st.pareto(alpha, scale=xm).rvs(num_jobs)
    
    completion_times = np.zeros(num_jobs)
    response_times = np.zeros(num_jobs)
    
    for i in range(1, num_jobs):
        if arrival_times[i] < completion_times[i - 1]:
            completion_times[i] = completion_times[i - 1] + service_times[i]
        else:
            completion_times[i] = arrival_times[i] + service_times[i]
        response_times[i] = completion_times[i] - arrival_times[i]
    
    mean_response_time = np.mean(response_times)
    return mean_response_time

# Simulate the SITA policy
for lambda_ in lambda_values:
    # Simulate single server for comparison
    mean_response_time_single = simulate_mg1(lambda_, 0.1, num_jobs)
    mean_response_times_single.append(mean_response_time_single)
    
    # Calculate theoretical response time for single server
    rho = lambda_ * E_S
    E_T_theoretical_single = E_S / (1 - rho) * (1 + C_S2 * rho / 2)
    theoretical_response_times_single.append(E_T_theoretical_single)
    
    # Split jobs based on cutoff
    interarrival_times = np.random.exponential(1 / lambda_, num_jobs)
    arrival_times = np.cumsum(interarrival_times)
    service_times = st.pareto(alpha, scale=xm).rvs(num_jobs)
    
    server1_service_times = service_times[service_times <= cutoff]
    server2_service_times = service_times[service_times > cutoff]
    
    mean_response_time_server1 = simulate_mg1(lambda_, mu1, len(server1_service_times))
    mean_response_time_server2 = simulate_mg1(lambda_, mu2, len(server2_service_times))
    
    mean_response_time_sita = (len(server1_service_times) * mean_response_time_server1 + 
                               len(server2_service_times) * mean_response_time_server2) / num_jobs
    mean_response_times_sita.append(mean_response_time_sita)
    
    # Calculate theoretical response time for SITA policy
    rho1 = lambda_ * np.mean(server1_service_times) / mu1
    rho2 = lambda_ * np.mean(server2_service_times) / mu2
    
    E_T_theoretical_sita = (
        np.mean(server1_service_times) / (1 - rho1) * (1 + C_S2 * rho1 / 2) +
        np.mean(server2_service_times) / (1 - rho2) * (1 + C_S2 * rho2 / 2)
    )
    theoretical_response_times_sita.append(E_T_theoretical_sita)

# Plot the results
for i, rho in enumerate(rho_values):
    print(f"For ρ = {rho}:")
    print(f"Single Server - Simulated E[T]: {mean_response_times_single[i]:.2f}, Theoretical E[T]: {theoretical_response_times_single[i]:.2f}")
    print(f"SITA Policy - Simulated E[T]: {mean_response_times_sita[i]:.2f}, Theoretical E[T]: {theoretical_response_times_sita[i]:.2f}")
    print(f"Improvement in Simulated E[T]: {(mean_response_times_single[i] - mean_response_times_sita[i]):.2f}")
    print(f"Improvement in Theoretical E[T]: {(theoretical_response_times_single[i] - theoretical_response_times_sita[i]):.2f}")

# Example plot (for ρ = 0.5 and ρ = 0.9)
plt.figure(figsize=(12, 6))
plt.plot(rho_values, mean_response_times_single, 'o-', label='Simulated Single Server')
plt.plot(rho_values, theoretical_response_times_single, 's--', label='Theoretical Single Server')
plt.plot(rho_values, mean_response_times_sita, 'o-', label='Simulated SITA Policy')
plt.plot(rho_values, theoretical_response_times_sita, 's--', label='Theoretical SITA Policy')
plt.xlabel('Utilization ρ')
plt.ylabel('Expected mean response time E[T] (seconds)')
plt.legend()
plt.title('Expected Mean Response Time E[T] vs. Utilization ρ')
plt.grid(True)
plt.show()
