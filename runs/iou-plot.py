import matplotlib.pyplot as plt

# 读取txt文件
data = []
with open("ious.txt", "r") as file:
    for line in file:
        data.append(float(line.strip()))

# 绘制曲线图
plt.scatter(range(len(data)), data, marker='o', s=10)
plt.xlabel('train_img')
plt.ylabel('ious')
plt.grid(True)
plt.savefig('ious.png')
plt.show()
