import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time

# 定数
g = 9.80  # 重力加速度
l = 1.0   # 振り子の長さ
m = 1.0   # 振り子の質量
theta0 = -np.pi / 2.0  # 初期の振り子の角度（90度にする）
omega0 = 0.0  # 初期の振り子の角速度
# F = 1.0 # 力
dt = 0.01


# 運動方程式（オイラー法）
def update_world(theta, omega, action):
    if action == 0:
        F = 0
    elif action == 1:
        F = 1.0
    elif action == 2:
        F = -1.0

    torque_f = F * l # 外力によるトルク
    torque_g = -m * g * l * np.sin(theta) # 重力によるトルク
    inertia = 1.0 * l**2 # 慣性モーメント
    torque = torque_f + torque_g # 合計のトルク
    aa = torque / inertia # 角加速度

    theta_new = theta + dt * omega
    omega_new = omega + dt * aa

    return theta_new, omega_new

# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.3  # ε-greedy法のε

# Qテーブルの初期化
num_theta_bins = 4
num_omega_bins = 4
num_actions = 3  # 行動数（例: 0, 1, 2）

Q = np.zeros((num_theta_bins, num_omega_bins, num_actions))

# 状態の離散化関数
def discretize_state(theta, omega):
    theta_bin = np.digitize(theta, np.linspace(-np.pi, np.pi, num_theta_bins + 1)) - 1
    omega_bin = np.digitize(omega, np.linspace(-2.0, 2.0, num_omega_bins + 1)) - 1

    # theta_binとomega_binが範囲内に収まるように調整
    theta_bin = max(0, min(num_theta_bins - 1, theta_bin))
    omega_bin = max(0, min(num_omega_bins - 1, omega_bin))

    return theta_bin, omega_bin

# リセット関数
def reset():
        theta = theta0
        omega = omega0

        return theta, omega

# ε-greedy法に基づく行動の選択
def select_action(theta_bin, omega_bin):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[theta_bin, omega_bin, :])


# Q学習のメイン関数
def q_learning(update_world):
    for epoch in range(50):  
        total_reward = 0
        theta, omega = reset()
        theta_bin, omega_bin = discretize_state(theta, omega)
        action = select_action(theta_bin, omega_bin)

        # CSVファイルの準備
        csv_file_path = f'single_pendulum_rl_data_epoch_{epoch + 1}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Time', 'Theta', 'Omega', 'Reward'])


            for i in range(1000):
                theta, omega = update_world(theta, omega, action)
                theta_bin, omega_bin = discretize_state(theta, omega)
                print(f'theta: {theta}, omega: {omega}')

                action = select_action(theta_bin, omega_bin)
                print(action)
            
                next_theta, next_omega = update_world(theta, omega, action)
                next_theta_bin, next_omega_bin = discretize_state(next_theta, next_omega)
                

                # Q値の更新
                reward_scale = 0.01
                reward = reward_scale * (theta**2 +  omega**2 + 0.1 * (next_theta - theta)**2 + 0.1 * (next_omega - omega)**2)  # 報酬の設計（例：振り子が安定している場合に報酬を与える）theta**2 + 0.1 * omega**2　→　theta**2 + omega**2
                
                total_reward = reward
                Q[theta_bin, omega_bin, action] += alpha * (reward + gamma * np.max(Q[next_theta_bin, next_omega_bin, :]) - Q[theta_bin, omega_bin, action])

                theta = next_theta
                omega = next_omega

                # CSVファイルにデータを保存
                csv_writer.writerow([i * dt, theta, omega, total_reward])



                print(f'Epoch: {epoch + 1}, Total Reward: {total_reward}')
                time.sleep(0.01)

        print(f'Data for epoch {epoch + 1} has been saved to {csv_file_path}')            

if __name__ == "__main__":
    # Q学習の実行
    q_learning(update_world)
