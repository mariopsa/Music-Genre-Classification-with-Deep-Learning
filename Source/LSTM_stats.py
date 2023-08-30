import json
import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt

# Load the results from the JSON file
results_file = "json_files/lstm_results.json"

with open(results_file, "r") as fp:
    results_data = json.load(fp)

# Extract test accuracies and errors from the results
test_accuracies = [result["test_accuracy"] for result in results_data]
test_errors = [result["test_error"] for result in results_data]

# Calculate statistical measures
accuracy_mean = np.mean(test_accuracies)
accuracy_std = np.std(test_accuracies)
error_mean = np.mean(test_errors)
error_std = np.std(test_errors)

# Additional statistical summary using describe from scipy.stats
accuracy_summary = describe(test_accuracies)
error_summary = describe(test_errors)

# Print the results
print("Test Accuracy Mean:", accuracy_mean)
print("Test Accuracy Standard Deviation:", accuracy_std)
print("Test Error Mean:", error_mean)
print("Test Error Standard Deviation:", error_std)

# Print additional summary using describe
print("\nAccuracy Summary:")
print("Nobs: ", accuracy_summary.nobs)
print("Min: ", accuracy_summary.minmax[0])
print("Max: ", accuracy_summary.minmax[1])
print("Mean: ", accuracy_summary.mean)
print("Variance: ", accuracy_summary.variance)
print("Standard Deviation: ", np.sqrt(accuracy_summary.variance))

print("\nError Summary:")
print("Nobs: ", error_summary.nobs)
print("Min: ", error_summary.minmax[0])
print("Max: ", error_summary.minmax[1])
print("Mean: ", error_summary.mean)
print("Variance: ", error_summary.variance)
print("Standard Deviation: ", np.sqrt(error_summary.variance))

# Create a box plot for accuracy
plt.figure(figsize=(8, 6))
plt.boxplot(test_accuracies)
plt.title("Test Accuracy Distribution (LSTM)")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(test_accuracies, bins=10, edgecolor='k', alpha=0.7)
plt.title("Test Accuracy Histogram (LSTM)")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(test_errors, test_accuracies)
plt.title("Traditional learning curve (LSTM)")
plt.xlabel("Test Error")
plt.ylabel("Test Accuracy")
plt.show()