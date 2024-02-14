#!/bin/bash

# Define your inputs here. For example:
inputs=("deberta_1e-6" "deberta_5e-6" "deberta_1e-5" "deberta_5e-5")

# Maximum number of concurrent jobs, equals to the number of GPUs
MAX_JOBS=4

# Function to check and wait for available job slot
check_jobs() {
  while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
    sleep 1
  done
}


# Loop through inputs and execute them
for i in "${!inputs[@]}"; do
  # Check if we need to wait for a job slot to become available
  check_jobs
  
  # Calculate GPU index: i % MAX_JOBS ensures cycling through GPUs 0 to MAX_JOBS-1
  gpu_index=$((i % MAX_JOBS))
  
  # Wait for 10 seconds before starting the next script
  sleep 10

  # Execute the script with CUDA_VISIBLE_DEVICES set for the specific GPU
  CUDA_VISIBLE_DEVICES=$gpu_index python myTrian2.py "${inputs[$i]}" &
done

# Wait for all background jobs to finish
wait

echo "All processes completed"