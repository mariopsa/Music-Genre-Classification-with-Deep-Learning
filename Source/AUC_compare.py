import json
import numpy as np
from scipy.stats import describe
import matplotlib.pyplot as plt
import pandas as pd

# Load the results from the JSON file
results_lstm = "lstm_auc.json"
results_cnn = "cnn_auc.json"

with open(results_lstm, "r") as fp:
    results_data_lstm = json.load(fp)
    
with open(results_cnn, "r") as fp:
    results_data_cnn = json.load(fp)
    
genres = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

auc_by_genre_lstm = {genre: [result[genre] for result in results_data_lstm] for genre in genres}
auc_by_genre_cnn = {genre: [result[genre] for result in results_data_cnn] for genre in genres}

mean_genre_lstm =[np.mean(auc_by_genre_lstm[genre]) for genre in genres]
mean_genre_cnn =[np.mean(auc_by_genre_cnn[genre]) for genre in genres]

std_genre_lstm =[np.std(auc_by_genre_lstm[genre]) for genre in genres]
std_genre_cnn =[np.std(auc_by_genre_cnn[genre]) for genre in genres]

auc_overall_lstm = [result["overall"] for result in results_data_lstm]
auc_overall_lstm_sum = describe(auc_overall_lstm)

auc_overall_cnn = [result["overall"] for result in results_data_cnn]
auc_overall_cnn_sum = describe(auc_overall_cnn)

# Create a DataFrame
data_mean = {
    "Genre": genres,
    "CNN": mean_genre_cnn,
    "LSTM": mean_genre_lstm
}

data_std = {
    "Genre": genres,
    "CNN": std_genre_cnn,
    "LSTM": std_genre_lstm
}

print("-*-*-*-*-* CNN -*-*-*-*-*")
print("Mean: ",auc_overall_cnn_sum.mean)    
print("Std.Deviation: ", np.sqrt(auc_overall_cnn_sum.variance))    
print("Min: ", auc_overall_cnn_sum.minmax[0])    
print("Max: ", auc_overall_cnn_sum.minmax[1])    
print("Variance: ", auc_overall_cnn_sum.variance)

print("-*-*-*-*-* LSTM -*-*-*-*-*")
print("Mean: ",auc_overall_lstm_sum.mean)    
print("Std.Deviation: ", np.sqrt(auc_overall_lstm_sum.variance))    
print("Min: ", auc_overall_lstm_sum.minmax[0])    
print("Max: ", auc_overall_lstm_sum.minmax[1])    
print("Variance: ", auc_overall_lstm_sum.variance)    

# Plot using pandas
df = pd.DataFrame(data_mean)
ax = df.plot(x="Genre", kind="bar", color=['cadetblue', '#FF9912'], width=0.6, edgecolor='black', figsize=(12, 6))
ax.set_xlabel("Musical genres")
ax.set_ylabel("AUC Score")
ax.set_ylim(0.85, 1)
ax.set_title("Mean AUC scores for each Music Genre by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot using pandas
ds = pd.DataFrame(data_std)
ax = ds.plot(x="Genre", kind="bar", color=['cadetblue', '#FF9912'], width=0.6, edgecolor='black', figsize=(12, 6))
ax.set_xlabel("Musical genres")
ax.set_ylabel("Standard deviation")
ax.set_ylim(0, 0.1)
ax.set_title("Standard Deviation AUC scores for each Music Genre by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()









