#! /usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ = "Lyon"
# Date = 2018/5/31

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------
# 数据可视化的重要性 Anscombe's quartet 安斯库姆四重奏
# ------------------------------------------------------------
anscombe_df = sns.load_dataset("anscombe")
print(anscombe_df.head())
anscombe_df.groupby("dataset").agg([np.mean, np.var])

sns.set(style="ticks")
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=anscombe_df, col_wrap=2, ci=None, palette="muted",
           size=4, scatter_kws={"s": 50, "alpha": 1})

sns.set(style="ticks")
sns.lmplot(x="x", y="y", col="dataset", hue="dataset",
           data=anscombe_df, col_wrap=2, ci=None, palette="muted",
           size=4, scatter_kws={"s": 50, "alpha": 1})
plt.show()

# ------------------------------------------------------------
# 基本图表的绘制及应用场景
# ------------------------------------------------------------
# --------------------------
#  1. Matplotlib架构
# --------------------------
mpl.get_backend()
# --------------------------
# 基本图表的绘制
# --------------------------
plt.plot(3, 2)
plt.show()
plt.plot(3, 2, '*')
plt.show()
# --------------------------
# 使用scripting 层绘制
# --------------------------
fig = Figure()
canvas = FigureCanvasAgg(fig)

ax = fig.add_subplot(111)
ax.plot(3, 2, '.')
canvas.print_png('test.png')

plt.figure()
plt.plot(3, 2, 'o')
plt.show()
# 设置坐标轴范围
plt.plot(3, 2, 'o')
ax = plt.gca()
ax.axis([0, 6, 0, 10])
plt.show()
# matplot 会自动用颜色区分不同的数据
plt.figure()
plt.plot(1.5, 1.5, 'o')
plt.plot(2, 2, '*')
plt.plot(2.5, 2.5, '*')
plt.show()

# --------------------------
# 散点图
# --------------------------
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = x
plt.figure()
plt.scatter(x, y)
plt.show()
# 改变颜色及大小
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = x
s = np.array([10, 20, 30, 30, 10, 40, 10, 20, 30])*9
colors = ["red"]*(len(x)-1)
colors.append("blue")
plt.figure()
plt.scatter(x, y, s=s, c=colors)
plt.show()
# 使用zip合并两个列表为一个新列表
# 新列表中的每个元素为对应位置上的元组
l1 = list(range(1, 6))
l2 = list(range(6, 11))
zip_generator = zip(l1, l2)
tuple_zip = list(zip_generator)
# 使用*进行对元组列表解包
x, y = zip(*tuple_zip)

plt.figure()
plt.scatter(x[:3], y[:3], c="red", label="Sample A")
plt.scatter(x[3:], y[3:], c="blue", label="Sample A")
plt.show()
# 添加坐标标签，标题
plt.figure()
plt.scatter(x[:3], y[:3], c="red", label="Sample A")
plt.scatter(x[3:], y[3:], c="blue", label="Sample A")
plt.xlabel("x label")
plt.ylabel("y label")
plt.title("Scatter Plot Example")
plt.show()
# Legend
plt.figure()
plt.scatter(x[:3], y[:3], c="red", label="Sample A")
plt.scatter(x[3:], y[3:], c="blue", label="Sample A")
plt.xlabel("x label")
plt.ylabel("y label")
plt.title("Scatter Plot Example")
plt.legend(loc=4, frameon=True, title="Legend")
plt.show()
# --------------------------
# 线图
# --------------------------
line_data = np.arange(1, 9)
quadratic_data = line_data**2
plt.figure()
plt.plot(line_data, "-o", quadratic_data, "-*")
plt.plot([22, 44, 66], '--r')
plt.show()

# 添加坐标轴标签及图例
line_data = np.arange(1, 9)
quadratic_data = line_data**2
plt.figure()
plt.plot(line_data, "-o", quadratic_data, "-*")
plt.plot([22, 44, 66], '--r')
plt.xlabel('x data')
plt.ylabel('y data')
plt.title('Line Chart Title')
plt.legend(['legend1', 'legend2', 'legend3'], title="Legend")
plt.show()

# 填充两个line间的区域
line_data = np.arange(1, 9)
quadratic_data = line_data**2
plt.figure()
plt.plot(line_data, "-o", quadratic_data, "-*")
# plt.gca().fill_between(np.arange(1, 9), line_data, quadratic_data, facecolors="green", alpha=0.3)
plt.gca().fill_between(range(len(line_data)), line_data, quadratic_data, facecolors="green", alpha=0.3)
plt.show()

# 绘制横轴为时间的线图
plt.figure()
observation_dates = np.arange("2017-10-11", '2017-10-19', dtype='datetime64[D]')
plt.plot(observation_dates, line_data, "-o",
         observation_dates, quadratic_data, "-*")
plt.show()

# 横轴并不是我们想要的结果
plt.figure()
observation_dates = np.arange('2017-10-11', '2017-10-19', dtype='datetime64[D]')
plt.plot(observation_dates, line_data, "-o",
         observation_dates, quadratic_data, "-*")
plt.xticks(rotation=45)
# x = plt.gca().xaxis
# for item in x.get_ticklabels():
#     item.set_rotation(45)
plt.show()

# 调整边界距离
plt.subplots_adjust(bottom=0.25)

# 对于学术制图，可在标题中包含latex语法
plt.figure()
observation_dates = np.arange('2017-10-11', '2017-10-19', dtype='datetime64[D]')
plt.plot(observation_dates, line_data, "-o",
         observation_dates, quadratic_data, "-*")
ax = plt.gca()
ax.set_title('Quadratic ($x^2$) vs. Linear ($x$)')
plt.show()

# --------------------------
# 柱状图
# --------------------------
plt.figure()
x_vals = list(range(len(line_data)))
plt.bar(x_vals, line_data, width=0.3)
plt.show()

# group bar chart
# 同一副图中添加新的柱状图
# 注意，为了不覆盖第一个柱状图，需要对x轴做偏移

plt.figure()
x_vals = list(range(len(line_data)))
plt.bar(x_vals, line_data, width=0.3)
plt.bar(x_vals, quadratic_data, width=0.3, bottom=line_data)
plt.show()

# 横向柱状图
plt.figure()
x_vals = list(range(len(line_data)))
plt.barh(x_vals, line_data, height=0.3)
plt.barh(x_vals, quadratic_data, height=0.3, left=line_data)
plt.show()


# ------------------------------------------------------------
# 数据分析常用图表的绘制
# ------------------------------------------------------------
# --------------------------
# 1. Subplots
# --------------------------
plt.figure()
plt.subplot(1, 2, 1)
linear_data = np.arange(1, 9)
plt.plot(linear_data, '-o')
# plt.show()

exponential_data = linear_data ** 2
plt.subplot(1, 2, 2)
plt.plot(exponential_data, '-x')
plt.show()


# 保证子图中坐标范围一致
plt.figure()
ax1 = plt.subplot(1, 2, 1)
linear_data = np.arange(1, 9)
plt.plot(linear_data, '-o')

exponential_data = linear_data ** 2
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
plt.plot(exponential_data, '-x')
plt.show()

# 多个坐标返回
plt.figure()
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True)
ax5.plot(exponential_data, '-')
plt.show()

# --------------------------
# 直方图
# --------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1, ax2, ax3, ax4]
for n in range(len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0, scale=1, size=sample_size)
    # 默认bin的个数为10
    axs[n].hist(sample)
    axs[n].set_title('n={}'.format(sample_size))
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1, ax2, ax3, ax4]
for n in range(len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0, scale=1, size=sample_size)
    # in的个数为1000
    axs[n].hist(sample, bins=1000)
    axs[n].set_title('n={}'.format(sample_size))
plt.show()

# --------------------------
# 使用gridspec和直方图绘制一个复杂分析图
# --------------------------
import matplotlib.gridspec as gridspec
x = np.random.random(size=10000)
y = np.random.normal(loc=1, scale=3, size=10000)
plt.figure()
gs = gridspec.GridSpec(3, 3)
top_hist = plt.subplot(gs[0, 1:])
left_hist = plt.subplot(gs[1:, 0])
main_hist = plt.subplot(gs[1:, 1:])
top_hist.hist(x, bins=100, normed=True)
left_hist.hist(y, bins=100, orientation="horizontal", normed=True)
left_hist.invert_xaxis()
main_hist.scatter(x, y)
plt.show()

# --------------------------
# 盒形图
# --------------------------
# 正态分布采样
normal_sample = np.random.normal(loc=0., scale=1., size=10000)
# 随机数采样
random_sample = np.random.random(size=10000)
# gamma分布采样
gamma_sample = np.random.gamma(2, size=10000)
df = pd.DataFrame({'normal': normal_sample,
                  'random': random_sample,
                  'gamma': gamma_sample})
df.describe()
plt.figure()
plt.boxplot(df["normal"],whis="range")
plt.show()

plt.figure()
# whis='range' 不显示离群值
plt.boxplot([df['normal'], df['random'], df['gamma']], whis='range')
plt.show()

plt.figure()
plt.boxplot([df['normal'], df['random'], df['gamma']])
plt.show()

# --------------------------
# 热图
# --------------------------
plt.figure()
y = np.random.normal(loc=0., scale=1., size=10000)
x = np.random.random(size=10000)
plt.hist2d(x, y, bins=25)
plt.colorbar()
plt.show()


# ------------------------------------------------------------
# Pandas及Seaborn绘图
# ------------------------------------------------------------
# 可用的绘图样式
plt.style.available
# 设置绘图样式
plt.style.use('seaborn-colorblind')

# --------------------------
# DataFrame绘图
# --------------------------
np.random.seed(100)
df = pd.DataFrame({'A': np.random.randn(365).cumsum(0), 'B': np.random.randn(365).cumsum(0) + 20,
                  'C': np.random.randn(365).cumsum(0) - 20}, index=pd.date_range('2017/1/1', periods=365))
df.head()


df.plot()
plt.show()
# plt.savefig("D:\img.jpg")

df.plot("A", "B", kind="scatter")
plt.show()

df.plot("A", "B", kind="scatter", c=df['B'], s=df['B'], colormap="viridis")
plt.show()

# 设置坐标为相同比例
ax = df.plot("A", "B", kind="scatter", c=df['B'], s=df['B'], colormap="viridis")
ax.set_aspect('equal')
plt.show()

df.plot(kind='box')
plt.show()

df.plot(kind='hist', alpha=0.7)
plt.show()

df.plot(kind='kde')
plt.show()

# --------------------------
# pandas.tools.plotting
# --------------------------
iris = pd.read_csv('iris.csv')
iris.head()

# 用于查看变量间的关系
pd.plotting.scatter_matrix(iris)
plt.show()

# 用于查看多遍量分布
plt.figure()
pd.plotting.parallel_coordinates(iris, 'Species')
plt.show()

# --------------------------
# Seaborn绘图
# --------------------------
import seaborn as sns
np.random.seed(100)
v1 = pd.Series(np.random.normal(0, 10, 1000), name='v1')
v2 = pd.Series(2 * v1 + np.random.normal(60, 15, 1000), name='v2')
# 通过matplotlib绘图
plt.figure()
plt.hist(v1, alpha=0.7, bins=np.arange(-50, 150, 5), label="v1")
plt.hist(v2, alpha=0.7, bins=np.arange(-50, 150,5), label="v2")
plt.legend()
plt.show()

plt.figure()
plt.hist([v1, v2], histtype='barstacked', normed=True)
# plt.hist([v1, v2], histtype='barstacked')
v3 = np.concatenate((v1, v2))
sns.kdeplot(v3)
plt.show()

# 使用seaborn绘图
plt.figure()
sns.distplot(v3)
plt.show()

# 使用seaborn绘图
plt.figure()
sns.jointplot(v1, v2, alpha=0.4)
plt.show()

# 使用seaborn绘图
plt.figure()
grid = sns.jointplot(v1, v2, alpha=0.4)
grid.ax_joint.set_aspect('equal')
plt.show()

plt.figure()
sns.jointplot(v1, v2, kind='hex')
plt.show()

plt.figure()
grid = sns.jointplot(v1, v2, kind='hex')
grid.ax_joint.set_aspect('equal')
plt.show()

plt.figure()
sns.jointplot(v1, v2, kind='kde')
plt.show()

plt.figure()
sns.pairplot(iris, hue='Species', diag_kind='kde')
# sns.pairplot(iris, diag_kind='kde')
plt.show()


# ------------------------------------------------------------
# 其他常用的可视化工具
# ------------------------------------------------------------
# --------------------------
# pyecharts
# --------------------------
# 柱状图交互式图例
# 参考 http://pyecharts.org/#/zh-cn/prepare
# pip install pyecharts
from pyecharts import Bar
comp_df = pd.read_csv('./comparison_result.csv', index_col='state')
comp_df
good_state_results = comp_df.iloc[0, :].values
heavy_state_results = comp_df.iloc[1, :].values
light_state_results = comp_df.iloc[2, :].values
medium_state_results = comp_df.iloc[3, :].values
labels = comp_df.index.values.tolist()
city_names = comp_df.columns.tolist()
bar = Bar("堆叠柱状图")
bar.add("良好", city_names, good_state_results, is_stack=True, xaxis_interval=0, xaxis_rotate=30)
bar.add('轻度污染', city_names, light_state_results, is_stack=True, xaxis_interval=0, xaxis_rotate=30)
bar.add('中度污染', city_names, medium_state_results, is_stack=True, xaxis_interval=0, xaxis_rotate=30)
bar.add('重度污染', city_names, heavy_state_results, is_stack=True, xaxis_interval=0, xaxis_rotate=30)
bar.render('./pyecharts/echarts_demo.html')








































