import numpy as np
import matplotlib.pyplot as plt
from vpython import *

# 常量
g = 9.81  # 重力加速度，单位 m/s^2
L = 0.1   # 摆长，单位米（10厘米）

# 定义微分方程组
def f(r, t):
    theta, omega = r
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

# 四阶龙格-库塔方法
def rk4_step(f, r, t, h):
    k1 = h * f(r, t)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(r + k3, t + h)
    return r + (k1 + 2*k2 + 2*k3 + k4) / 6

# 初始条件
t0 = 0.0
t_end = 10.0
h = 0.01
t_points = np.arange(t0, t_end, h)
theta0 = np.radians(179)  # 初始角度（179度），转换为弧度
omega0 = 0.0              # 初始角速度
r = np.array([theta0, omega0])

# 用于存储解的数组
theta_points = []
omega_points = []

# 求解方程
for t in t_points:
    theta_points.append(r[0])
    omega_points.append(r[1])
    r = rk4_step(f, r, t, h)

# 将theta从弧度转换为度，便于理解
theta_points_deg = np.degrees(theta_points)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t_points, theta_points_deg, label='omega (θ)')
plt.xlabel('time (s)')
plt.ylabel('omega (°)')
plt.title('the motion of a nonlinear pendulum')
plt.legend()
plt.grid(True)
plt.show()

# 动画
# VPython 可视化
scene = canvas(title='Nonlinear Pendulum', width=800, height=600, center=vector(0, -L/2, 0), background=color.white)

# 创建摆的杆和球
pivot = vector(0, 0, 0)
ball = sphere(pos=vector(L * np.sin(theta0), -L * np.cos(theta0), 0), radius=0.01, color=color.red)
rod = cylinder(pos=pivot, axis=ball.pos - pivot, radius=0.005, color=color.blue)

# 动画
for theta in theta_points:
    rate(100)  # 控制动画速度
    ball.pos = vector(L * np.sin(theta), -L * np.cos(theta), 0)
    rod.axis = ball.pos - pivot