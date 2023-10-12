import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.log(1 + x * x)


x_values = np.linspace(0.01, 2, 400)
valid_indices = np.where(1 + 1.5*1.5 * x_values > 0)
x_valid = x_values[valid_indices]

# 计算y值
y_values = func(x_valid)
# 绘制图像
plt.plot(x_valid, y_values, label=r'$\frac{\log(1+x)}{\log(1+1.5^2x)}$')
plt.title(r'Graph of $\frac{\log(1+x)}{\log(1+1.5^2x)}$')
plt.xlabel('x')
plt.ylabel('f(x)')

# 添加图例和网格
plt.legend()
plt.grid(True)

# 显示图像
plt.savefig('graph.png')
