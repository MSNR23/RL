import numpy as np
import csv
import time

# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.1  # ε-greedy法のε

# Qテーブルの初期化
num_theta_bins = 20
num_omega_bins = 20
num_actions = 3  # 行動数（例: -1、0、1）

Q = np.zeros((num_theta_bins, num_omega_bins, num_actions))


# 状態の離散化関数
def discretize_state(theta, omega):
    theta_bin = np.digitize(theta, np.linspace(-np.pi, np.pi, num_theta_bins + 1)) - 1
    omega_bin = np.digitize(omega, np.linspace(-2.0, 2.0, num_omega_bins + 1)) - 1

    # theta_binとomega_binが範囲内に収まるように調整
    theta_bin = max(0, min(num_theta_bins - 1, theta_bin))
    omega_bin = max(0, min(num_omega_bins - 1, omega_bin))

    return theta_bin, omega_bin

# ε-greedy法に基づく行動の選択
def select_action(theta_bin, omega_bin):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[theta_bin, omega_bin, :])



# Q学習のメイン関数
def q_learning(s_values):
    # global epoch
    # global total_reward
    for epoch in range(150):  
        total_reward = 0
        for i in range(len(s_values) - 1):
            # データが角度と角速度の2つの値からなることを確認
            if len(s_values[i]) != 2:
                raise ValueError("Invalid data format. Expected 2 values (theta and omega).")

            theta, omega = s_values[i]
            theta_bin, omega_bin = discretize_state(theta, omega)

            action = select_action(theta_bin, omega_bin)
            next_theta, next_omega = s_values[i + 1]

            next_theta_bin, next_omega_bin = discretize_state(next_theta, next_omega)

            # Q値の更新
            reward_scale = 0.01
            reward = reward_scale * (theta**2 +  omega**2 + 0.1 * (next_theta - theta)**2 + 0.1 * (next_omega - omega)**2)  # 報酬の設計（例：振り子が安定している場合に報酬を与える）theta**2 + 0.1 * omega**2　→　theta**2 + omega**2
            
            total_reward = reward
            Q[theta_bin, omega_bin, action] += alpha * (reward + gamma * np.max(Q[next_theta_bin, next_omega_bin, :]) - Q[theta_bin, omega_bin, action])

            print(f'Epoch: {epoch + 1}, Total Reward: {total_reward}')
            time.sleep(1)
            
        # print(f'Epoch: {epoch + 1}')
        # time.sleep(1)


# データをCSVファイルから読み込む関数
def load_data(csv_file_path):
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # ヘッダー行をスキップ
        data = [list(map(float, row[1:])) for row in csv_reader]  # 2番目以降の列を取得
    return np.array(data)

if __name__ == "__main__":
    # CSVファイルからデータを読み込む
    csv_file_path = 'single_pendulum_data.csv'
    simulation_data = load_data(csv_file_path)

    # Q学習の実行
    q_learning(simulation_data)