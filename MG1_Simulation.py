import numpy as np
import matplotlib.pyplot as plt

# Define parameters for log-normal distribution
mean_service_time = 10  # seconds
scv = 16                # Squared coefficient of variation
variance_service_time = scv * (mean_service_time ** 2)       # variance = E[s^2] - E[s]^2

E_S_sqrd = variance_service_time + mean_service_time ** 2    # E[s^2] = variance + E[s]^2

# Calculate log-normal parameters mu and sigma
sigma_squared = np.log(variance_service_time / (mean_service_time ** 2) + 1)
mu = np.log(mean_service_time) - sigma_squared / 2



# Generate service times from log-normal distribution
def generate_service_times(n, mu, sigma_squared):
    sigma = np.sqrt(sigma_squared)
    return np.random.lognormal(mu, sigma, n)

# Simulate the M/G/1/FCFS queue
def simulate_queue(lambda_rate, mu, sigma_squared, num_jobs=1000000):
    inter_arrival_times = np.random.exponential(scale=1/lambda_rate, size=num_jobs) # new job will either start when it arrives in the system, or when the previous job is finished 
    arrival_times = np.cumsum(inter_arrival_times)                                  # finish time is the time that the process starts to be processed plus the duration of the processing
    service_times = generate_service_times(num_jobs, mu, sigma_squared)             # total response time is the finish time minus the time of the arrival
    
    
    start_times = np.zeros(num_jobs)
    finish_times = np.zeros(num_jobs)
    response_times = np.zeros(num_jobs)
    
    for i in range(1, num_jobs):
        start_times[i] = max(arrival_times[i], finish_times[i-1])
        finish_times[i] = start_times[i] + service_times[i]
        response_times[i] = finish_times[i] - arrival_times[i]
             
    return np.mean(response_times)  # Return the mean of the response time E[T]

# Theoretical mean response time
# This function is used to calculate the theoretical responese time (to compare with the simulated one)
def theoretical_mean_response(lambda_rate):
    rho = lambda_rate * mean_service_time
    E_S_sqrd = variance_service_time + mean_service_time ** 2 # In theory, ρ = λ * Ε[S] where E[S] = 10 as defined by the problem
    E_T = mean_service_time + (lambda_rate * E_S_sqrd) / (2 * (1 - rho)) # Ε[Τ] = E[s] + λE[s^2]/(2(1-ρ))
                                                                         # This formula accounts for both the average service time
                                                                         # and the variability in service time (variance). 
                    
    return E_T

# Define range of lambda values
# we know from theory that λ = ρ / Ε[s]
rho = np.arange(0.1, 1.0, 0.1)
lambda_vals = rho / mean_service_time

# Initialize lists to store results
simulated_mean_responses = []
theoretical_mean_responses = []

# Run simulations for each λ value
for lambda_rate in lambda_vals:
    simulated_mean_responses.append(simulate_queue(lambda_rate, mu, sigma_squared))
    theoretical_mean_responses.append(theoretical_mean_response(lambda_rate))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(lambda_vals, simulated_mean_responses, label='Simulated Mean Response Time', marker='o')
plt.plot(lambda_vals, theoretical_mean_responses, label='Theoretical Mean Response Time', marker='x')
plt.xlabel('Arrival Rate (lambda)')
plt.ylabel('Mean Response Time (E[T])')
plt.title('M/G/1/FCFS Queue Mean Response Time with Log-Normal Distribution')
plt.legend()
plt.grid(True)
plt.show()
