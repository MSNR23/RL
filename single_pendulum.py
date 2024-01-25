import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# 定数
g = 9.80  # 重力加速度
l = 1.0   # 振り子の長さ
theta0 = np.pi / 2.0  # 初期の振り子の角度（90度にする）
omega0 = 0.0  # 初期の振り子の角速度

# # 時間パラメータ
# dt = 0.01  # 時間刻み幅
# t_end = 10.0  # シミュレーション終了時間

# 運動方程式（オイラー法）
def euler_method(theta, omega, dt):
    theta_new = theta + dt * omega
    omega_new = omega - (g / l) * np.sin(theta) * dt
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

    # シミュレーション実行
    for t in t_values:
        theta_values.append(theta)
        omega_values.append(omega)
        theta, omega = euler_method(theta, omega, dt)

    # CSVファイルにデータを保存
    csv_file_path = 'single_pendulum_data.csv'
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
    animation_file_path = 'single_pendulum_animation.gif'
    ani.save(animation_file_path, writer='pillow', fps=30)
    print(f'Animation has been saved to {animation_file_path}')

    # アニメーションの表示
    plt.show()

if __name__ == "__main__":
    main()