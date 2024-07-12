import numpy as np
import matplotlib.pyplot as plt

# Define parameters for Pareto distribution
mean_service_time = 10  # seconds
scv = 16                # Squared coefficient of variation
variance_service_time = scv * mean_service_time ** 2

# We are using the pareto distribution as the job execution process
def calculate_pareto_parameters(mean, variance):
    # using the given mean service time, squared coefficient
    # of variation and the calculated variance, we find the paretto parameters
    alpha = (2 * mean ** 2 + variance) / (variance - mean ** 2)
    xm = (alpha - 1) * mean / alpha
    return xm, alpha

# Generate service times from Pareto distribution
def generate_service_times(n, xm, alpha):
    return (np.random.pareto(alpha, n) + 1) * xm

# Simulate the M/G/1/FCFS queue
def simulate_queue(lambda_rate, xm ,alpha, num_jobs=100000):
    # the arrival processes follows the poisson distribution
    # so the time between each arrival is exponentially distributed
    inter_arrival_times = np.random.exponential(scale=1/lambda_rate, size=num_jobs)
    arrival_times = np.cumsum(inter_arrival_times)

    # create the service time based on the distribution that we have selected (pareto)
    service_times = generate_service_times(num_jobs, xm, alpha) 
    
    start_times = np.zeros(num_jobs)
    finish_times = np.zeros(num_jobs)
    response_times = np.zeros(num_jobs)
    
    for i in range(1, num_jobs):
        start_times[i] = max(arrival_times[i], finish_times[i-1]) # new job will either start when it arrives in the system, or when the previous job is finished 
        finish_times[i] = start_times[i] + service_times[i]       # finish time is the time that the process starts to be processed plus the duration of the processing
        response_times[i] = finish_times[i] - arrival_times[i]    # total response time is the finish time minus the time of the arrival
    
    return np.mean(response_times)   # return the mean of the response time E[T]
 
# Theoretical mean response time
# This function is used to calculate the theoretical responese time (to compare with the simulated one)
def theoretical_mean_response(lambda_rate):
    rho = lambda_rate * mean_service_time # In theory, ρ = λ * Ε[S] where E[S] = 10 as defined by the problem
    E_T = mean_service_time * (1 + (lambda_rate * variance_service_time) / (2 * (1 - rho))) # Ε[Τ] = μ[1+(λ*σ^2)/(2(1-ρ))]
                                                                                            # This formula accounts for both the average service time
                                                                                            # and the variability in service time (variance). 
    return E_T



# Calculate the pareto parameters
xm, alpha = calculate_pareto_parameters(mean_service_time, variance_service_time)

# Define range of lambda values
# we know from theory that λ = ρ / Ε[s]
rho = np.arange(0.1, 1.0, 0.1)  
lambda_vals = rho / mean_service_time
print(lambda_vals)
simulated_mean_responses = []
theoretical_mean_responses = []


# Run simulations for each λ value
for lambda_rate in lambda_vals:
    simulated_mean_responses.append(simulate_queue(lambda_rate,xm,alpha))
    theoretical_mean_responses.append(theoretical_mean_response(lambda_rate))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(lambda_vals, simulated_mean_responses, label='Simulated Mean Response Time', marker='o')
plt.plot(lambda_vals, theoretical_mean_responses, label='Theoretical Mean Response Time', marker='x')
plt.xlabel('Arrival Rate (lambda)')
plt.ylabel('Mean Response Time (E[T])')
plt.title('M/G/1/FCFS Queue Mean Response Time with Pareto Distribution')
plt.legend()
plt.grid(True)
plt.show()
