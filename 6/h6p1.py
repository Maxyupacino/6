import numpy as np
import matplotlib.pyplot as plt

# 定义常数
alpha = 1.0
beta = 0.5
gamma = 0.5
delta = 2

# 定义微分方程组
def f(r, t):
    x, y = r
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    return np.array([dxdt, dydt])

# 四阶龙格-库塔方法
def rk4_step(f, r, t, h):
    k1 = h * f(r, t)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(r + k3, t + h)
    return r + (k1 + 2*k2 + 2*k3 + k4) / 6

# 初始条件
t0 = 0.0
t_end = 30.0
h = 0.01
t_points = np.arange(t0, t_end, h)
x0 = 2.0
y0 = 2.0
r = np.array([x0, y0])

# 用于存储解的数组
x_points = []
y_points = []

# 求解方程
for t in t_points:
    x_points.append(r[0])
    y_points.append(r[1])
    r = rk4_step(f, r, t, h)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(t_points, x_points, label='rabbit (x)')
plt.plot(t_points, y_points, label='fox (y)')
plt.xlabel('time')
plt.ylabel('number')
plt.title('Lotka-Volterra model')
plt.legend()
plt.grid(True)
plt.show()