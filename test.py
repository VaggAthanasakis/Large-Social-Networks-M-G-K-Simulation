import numpy as np
import matplotlib.pyplot as plt

# Define parameters
mean_service_time = 10  # seconds
scv = 16  # Squared coefficient of variation
variance_service_time = scv * (mean_service_time ** 2)

# Calculate log-normal parameters mu and sigma
sigma_squared = np.log(variance_service_time / (mean_service_time ** 2) + 1)
mu = np.log(mean_service_time) - sigma_squared / 2

# Generate service times from log-normal distribution
def generate_service_times(n, mu, sigma_squared):
    sigma = np.sqrt(sigma_squared)
    return np.random.lognormal(mu, sigma, n)

# Simulate the SITA policy with two servers
def simulate_sita_queue(lambda_rate, mu1, mu2, cutoff, num_jobs=1000000):
    inter_arrival_times = np.random.exponential(scale=1/lambda_rate, size=num_jobs)
    arrival_times = np.cumsum(inter_arrival_times)

    # Separate job sizes into small and large based on cutoff
    small_job_indices = []
    large_job_indices = []

    for i in range(num_jobs):
        if np.random.lognormal(mu, np.sqrt(sigma_squared)) <= cutoff:
            small_job_indices.append(i)
        else:
            large_job_indices.append(i)

    # Generate service times for each server
    service_times_1 = generate_service_times(len(small_job_indices), mu, sigma_squared)
    service_times_2 = generate_service_times(len(large_job_indices), mu, sigma_squared)
    
    start_times_1 = np.zeros(len(small_job_indices))
    finish_times_1 = np.zeros(len(small_job_indices))
    response_times_1 = np.zeros(len(small_job_indices))

    start_times_2 = np.zeros(len(large_job_indices))
    finish_times_2 = np.zeros(len(large_job_indices))
    response_times_2 = np.zeros(len(large_job_indices))

    # Process small jobs
    for i in range(1, len(small_job_indices)):
        idx = small_job_indices[i]
        start_times_1[i] = max(arrival_times[idx], finish_times_1[i-1])
        finish_times_1[i] = start_times_1[i] + service_times_1[i]
        response_times_1[i] = finish_times_1[i] - arrival_times[idx]

    # Process large jobs
    for i in range(1, len(large_job_indices)):
        idx = large_job_indices[i]
        start_times_2[i] = max(arrival_times[idx], finish_times_2[i-1])
        finish_times_2[i] = start_times_2[i] + service_times_2[i]
        response_times_2[i] = finish_times_2[i] - arrival_times[idx]

    # Combine response times
    combined_response_times = np.concatenate((response_times_1, response_times_2))
    
    return np.mean(combined_response_times)

# Theoretical mean response time for SITA
def theoretical_mean_response_sita(lambda_rate, mu1, mu2, cutoff):
    rho1 = lambda_rate * (1 / mu1) * (1 - scv / (scv + 1))
    rho2 = lambda_rate * (1 / mu2) * (scv / (scv + 1))
    
    E_S1_sqrd = (variance_service_time / (scv + 1)) + (mean_service_time * (1 - scv / (scv + 1))) ** 2
    E_S2_sqrd = (variance_service_time * (scv / (scv + 1))) + (mean_service_time * (scv / (scv + 1))) ** 2
    
    E_T1 = (1 / mu1) + (lambda_rate * E_S1_sqrd) / (2 * (1 - rho1))
    E_T2 = (1 / mu2) + (lambda_rate * E_S2_sqrd) / (2 * (1 - rho2))
    
    return E_T1, E_T2

# Define cutoff point and server capacities
cutoff = mean_service_time * np.sqrt(2)
mu1, mu2 = 0.05, 0.05

# Define range of lambda values
rho = np.array([0.5, 0.9])
lambda_vals = rho / mean_service_time

# Initialize lists to store results
simulated_mean_responses_sita = []
theoretical_mean_responses_sita = []

# Run simulations for each λ value
for lambda_rate in lambda_vals:
    simulated_mean_responses_sita.append(simulate_sita_queue(lambda_rate, mu1, mu2, cutoff))
    E_T1, E_T2 = theoretical_mean_response_sita(lambda_rate, mu1, mu2, cutoff)
    theoretical_mean_responses_sita.append((E_T1 + E_T2) / 2)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(lambda_vals, simulated_mean_responses_sita, label='Simulated Mean Response Time SITA', marker='o')
plt.plot(lambda_vals, theoretical_mean_responses_sita, label='Theoretical Mean Response Time SITA', marker='x')
plt.xlabel('Arrival Rate (lambda)')
plt.ylabel('Mean Response Time (E[T])')
plt.title('M/G/1/FCFS Queue Mean Response Time with SITA Policy and Log-Normal Distribution')
plt.legend()
plt.grid(True)
plt.show()
