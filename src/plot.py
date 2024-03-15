import pandas as pd
import matplotlib.pyplot as plt

# Plot the average success ratios of the DQN algorithm over n timesteps and m test iterations
def plot_avg_success_ratios(df):
    dqn_data = df[df['Algorithm'] == 'DQN']

    plt.figure(figsize=(10, 6))

    algorithm_data = dqn_data[dqn_data['Algorithm'] == 'DQN']
    plt.scatter(algorithm_data['Timesteps'], algorithm_data['Success ratio'], label='DQN - Individual', marker='x', color='red')
    plt.plot(algorithm_data.groupby('Timesteps')['Success ratio'].mean(), label='DQN - Average', marker='o', linestyle='solid', color='blue')

    plt.title('Average Success Ratios - DQN')
    plt.xlabel('Timesteps')
    plt.ylabel('Success Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = '../metrics.csv'
    df = pd.read_csv(file_path)
    plot_avg_success_ratios(df)