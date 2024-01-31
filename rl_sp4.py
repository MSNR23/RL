import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import time

# # Q学習のパラメータ
# alpha = 0.1  # 学習率
# gamma = 0.9  # 割引率
# epsilon = 0.1  # ε-greedy法のε

# # Qテーブルの初期化
# num_theta_bins = 4
# num_omega_bins = 4
# num_actions = 3  # 行動数（例: -1、0、1）

# Q = np.zeros((num_theta_bins, num_omega_bins, num_actions))

# 定数
g = 9.80  # 重力加速度
l = 1.0   # 振り子の長さ
m = 1.0   # 振り子の質量
theta0 = np.pi / 2.0  # 初期の振り子の角度（90度にする）
omega0 = 0.0  # 初期の振り子の角速度
F = 1.0 # 力


# 運動方程式（オイラー法）
def update_world(theta, omega, F, torque, dt):
    # F = 1.0 # 力
    torque_f = F * l # 外力によるトルク
    torque_g = -m * g * l * np.sin(theta) # 重力によるトルク
    inertia = 1.0 * l**2 # 慣性モーメント
    torque = torque_f + torque_g # 合計のトルク
    aa = torque / inertia # 角加速度

    theta_new = theta + dt * omega
    omega_new = omega + dt * aa
    return theta_new, omega_new

def main():
    # シミュレーションの初期化
    # 時間パラメータ
    dt = 0.01  # 時間刻み幅
    t_end = 10.0  # シミュレーション終了時間
    t_values = np.arange(0, t_end, dt)
    theta_values = []
    omega_values = []

    # 初期条件
    theta = theta0
    omega = omega0

    torque_f = F * l # 外力によるトルク
    torque_g = -m * g * l * np.sin(theta) # 重力によるトルク
    torque = torque_f + torque_g # 合計のトルク


    # シミュレーション実行
    for t in t_values:
        theta_values.append(theta)
        omega_values.append(omega)
        theta, omega = update_world(theta, omega, F, torque, dt)

    # CSVファイルにデータを保存
    csv_file_path = 'single_pendulum_rl_data.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time', 'Theta', 'Omega'])
        for i in range(len(t_values)):
            csv_writer.writerow([t_values[i], theta_values[i], omega_values[i]])
    
    print(f'Data has been saved to {csv_file_path}')

    # アニメーションの準備
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)

    # アニメーションの更新関数
    def update(frame):
        x = l * np.sin(theta_values[frame])
        y = -l * np.cos(theta_values[frame])
        line.set_data([0, x], [0, y])
        return [line]

    # アニメーションの作成
    ani = FuncAnimation(fig, update, frames=len(t_values), blit=True)

    # アニメーションの保存（GIF形式）
    animation_file_path = 'single_pendulum_rl_animation.gif'
    ani.save(animation_file_path, writer='pillow', fps=30)
    print(f'Animation has been saved to {animation_file_path}')

    # アニメーションの表示
    plt.show()

# Q学習のパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.1  # ε-greedy法のε

# Qテーブルの初期化
num_theta_bins = 4
num_omega_bins = 4
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
def q_learning(s_values, dt):
    for epoch in range(50):  
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

            # # 振り子の挙動に対する報酬を表示
            # print(f'Time: {i * dt}, Theta: {theta}, Omega: {omega}, Reward: {reward}')
            # time.sleep(0.3)

            print(f'Epoch: {epoch + 1}, Total Reward: {total_reward}')
            time.sleep(0.3)
            


if __name__ == "__main__":
    # mainプログラムの実行
    main()
    # シミュレーションのデータをCSVファイルから読み込む
    csv_file_path = 'single_pendulum_rl_data.csv'
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)
    # データの最初の列が時間なので、2列目と3列目を取得してs_valuesに格納
    s_values = data[:, 1:3]
    # 時間刻み幅を取得
    dt = 0.01
    # Q学習の実行
    q_learning(s_values, dt)