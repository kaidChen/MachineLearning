# 训练集
# 每个样本点有3个分量 (x0,x1,x2)
x = [(1, 0., 3), (1, 1., 3), (1, 2., 3), (1, 3., 2), (1, 4., 4), (1, 1, 1)]
# y[i] 样本点对应的输出
y = [14, 17, 20, 19, 30, 9]

# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
epsilon = 0.00001

# 学习率
alpha = 0.01
error1 = 0
error0 = 0
cnt = 0
m = len(x)

# 初始化参数
theta0 = 0
theta1 = 0
theta2 = 0

while True:
    cnt += 1

    # 参数迭代计算
    for i in range(m):
        # 拟合函数为 y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
        # 计算残差
        diff = (theta0 + theta1 * x[i][1] + theta2 * x[i][2]) - y[i]

        # 梯度 = diff[0] * x[i][j]
        theta0 -= alpha * diff * x[i][0]
        theta1 -= alpha * diff * x[i][1]
        theta2 -= alpha * diff * x[i][2]

    # 计算损失函数
    error1 = 0
    for lp in range(len(x)):
        error1 += (y[lp] - (theta0 + theta1 * x[lp][1] + theta2 * x[lp][2])) ** 2 / 2

    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1

    print(' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (theta0, theta1, theta2, error1))
print('Done: theta0 : %f, theta1 : %f, theta2 : %f' % (theta0, theta1, theta2))
print('迭代次数: %d' % cnt)
