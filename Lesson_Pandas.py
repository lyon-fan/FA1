# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------------------------------------
# 基本数据对象及操作
# ------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------
# 0.1. Series
# ------------------------------------------------------
import numpy as np
import pandas as pd

# 创建Series
countries = ["中国", "美国", "日本"]
s_countries = pd.Series(countries)
print(type(s_countries))
print(s_countries)

numbers = [1, 2, 3, 4]
print(pd.Series(numbers))

country_dict = {"CH": "中国", "US": "美国", "AU": "日本"}
s_country_dict = pd.Series(country_dict)
print(s_country_dict)
# 给索引命名
s_country_dict.index.name = "Code"
print(s_country_dict)
# 给数据命名
s_country_dict.name = "Country"
print(s_country_dict)
print(s_country_dict.values)
print(s_country_dict.index)
print(s_country_dict.name)
# 处理缺失数据
countries = ['中国', '美国', '澳大利亚', None]
print(pd.Series(countries))
numbers = [4, 5, 6, None]
print(pd.Series(numbers))
# Series 索引
country_dicts = {'CH': '中国',
                 'US': '美国',
                 'AU': '澳大利亚'}
s_country_dicts = pd.Series(country_dicts)
print(s_country_dicts)
# 通过索引判断数据是存在
# Series也可看作定长、有序的字典
print('CH' in s_country_dicts)

print('iloc:', s_country_dicts[0])
print("loc:", s_country_dicts.loc["US"])
print("[]:", s_country_dicts["US"])

print('iloc:\n', s_country_dicts.iloc[[0, 2]])
print()
print('loc:\n', s_country_dicts.loc[["AU", "CH"]])
# 向量化操作
s = pd.Series(np.random.randint(0, 1000, 10000))
s.head()
len(s)

total = 0
for i in s:
    total += i
total = np.sum(s)

s = pd.Series(np.random.randint(0, 1000, 10000))
for label, value in s.iteritems():
    s.loc[label] = value + 2
# ------------------------------------------------------
# 0.2. DataFrame
# ------------------------------------------------------
country1 = pd.Series({'Name': '中国',
                      'Language': 'Chinese',
                      'Area': '9.597M km2',
                      'Happiness Rank': 79})

country2 = pd.Series({'Name': '美国',
                      'Language': 'English (US)',
                      'Area': '9.834M km2',
                      'Happiness Rank': 14})

country3 = pd.Series({'Name': '澳大利亚',
                      'Language': 'English (AU)',
                      'Area': '7.692M km2',
                      'Happiness Rank': 9})
# df = pd.DataFrame([country1, country2, country3])
df = pd.DataFrame([country1, country2, country3], index=['CH', 'US', 'AU'])

# 添加数据
# 如果个数小于要求的个数，会自动进行“广播”操作
# 如果大于要求的个数，会报错
df["location"] = "地球"
df['Region'] = ['亚洲', '北美洲', '大洋洲']

# Dataframe索引
df.index.name = "index"
print(df)
print(type(df))
print(df.loc["CH"])
print(type(df.loc["CH"]))
print(df.iloc[1])
print(df['Area'])
print(df[['Area', "Name"]])

# 混合索引
# 注意写法上的区别
print('先取出列，再取行：')
print(df['Area']['CH'])
print(df['Area'].loc['CH'])
print(df['Area'].iloc[0])
print('先取出行，再取列：')
print(df.loc["CH"]["Area"])
print(df.iloc[1]["Area"])
# 转换行和列
print(df.T)
# 删除行数据
print(df.drop(['CH']))
# 注意drop操作只是将修改后的数据copy一份，而不会对原始数据进行修改
print(df)
# 如果使用了inplace=True，会在原始数据上进行修改，同时不会返回一个copy
print(df.drop(["CH"], inplace=True))
print(df)
#  如果需要删除列，需要指定axis=1
print(df.drop(["Area"], axis=1))
# 也可直接使用del关键字 del就直接删除了
del df["Name"]
# DataFrame的操作与加载
df['Happiness Rank']
# 注意从DataFrame中取出的数据进行操作后，会对原始数据产生影响
ranks = df['Happiness Rank']
ranks += 2
print(ranks)
print(df)
# 注意从DataFrame中取出的数据进行操作后，会对原始数据产生影响
# 安全的操作是使用copy()
ranks = df['Happiness Rank'].copy()
ranks += 2
print(ranks)
print(df)

# 加载csv文件数据
report_2015_df = pd.read_csv('./2015.csv')
print('2015年数据预览：')
report_2015_df.head()
print(report_2015_df.info())
# ------------------------------------------------------
# 0.3. 索引
# ------------------------------------------------------
# 使用index_col指定索引列
# 使用usecols指定需要读取的列
report_2016_df = pd.read_csv('./2016.csv',
                             index_col="Country", usecols=["Country", 'Happiness Rank', 'Happiness Score', 'Region'])
# 数据预览
report_2016_df.head()
print('列名(column)：', report_2016_df.columns)
print('行名(index)：', report_2016_df.index)
# 注意index是不可变的
report_2016_df.index[0] = '丹麦'
# 重置index
# 注意inplace加与不加的区别 不加会返回重新排序后的df 加会返回一个空
report_2016_df.reindex(inplace=True)
report_2016_df.head()
# 重命名列名
report_2016_df = report_2016_df.rename(
        columns={'Region': '地区', 'Happiness Rank': '排名', 'Happiness Score': '幸福指数'})
report_2016_df.head()
# 重命名列名，注意inplace的使用
report_2016_df.rename(columns={'Region': '地区', 'Happiness Rank': '排名', 'Happiness Score': '幸福指数'},
                      inplace=True)
report_2016_df.head()
# ------------------------------------------------------
# 0.4. Boolean Mask
# ------------------------------------------------------
report_2016_df.head()
# 过滤 Western Europe 地区的国家
# only_western_europe = report_2016_df['地区'] == 'Western Europe'
report_2016_df[report_2016_df['地区'] == 'Western Europe']
only_western_europe_10 = (report_2016_df["地区"] == 'Western Europe') & (report_2016_df["排名"] > 10)
report_2016_df[(report_2016_df['地区'] == 'Western Europe') & (report_2016_df["排名"] > 10)]

# ------------------------------------------------------
# 0.5. 层级索引
# ------------------------------------------------------
report_2015_df.head()
report_2015_df2 = report_2015_df.set_index(['Region', 'Country'])
report_2015_df2.head(20)
# level0 索引
report_2015_df2.loc["Western Europe"]
# 两层索引
report_2015_df2.loc['Western Europe', 'Switzerland']
# 交换分层顺序
report_2015_df2.swaplevel()
# 排序分层
report_2015_df2.sort_index(level=0)
# ------------------------------------------------------------------------------------------------------------
# 数据清洗
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------
# 处理缺失数据
# ------------------------------------------------------
log_data = pd.read_csv('./log.csv')
print(log_data)
# 判断数据缺失
log_data.isnull()
log_data['paused'].isnull()
# 取出volume不为空的数据
test = log_data[log_data["volume"].notnull()]
print(test)
log_data.set_index(['time', 'user'], inplace=True)
log_data.sort_index(inplace=True)
print(log_data)

log_data.fillna(0)
log_data.dropna()
log_data.ffill()
log_data.bfill()
# ------------------------------------------------------
# 数据变形
# ------------------------------------------------------
# 处理重复数据
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                     'k2': [1, 1, 2, 3, 3, 4, 4]})
# 判断数据是否重复
data.duplicated()
# 如果duplicated方法和drop_duplicates方法中没有设置参数，则这两个方法默认会判断全部咧，如果在这两个方法中加入了指定的属性名（或者称为列名）
data2 = data.drop_duplicates()

data['v1'] = range(7)
print(data)

# 去除指定列的重复数据 只留下第一个
data.drop_duplicates(['k1'])
# 去除指定列的重复数据 只留下最后一个
data.drop_duplicates(['k1', 'k2'], keep='last')

# 使用函数或map转化数据
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef', 'Bacon', 'pastrami',
                              'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
print(data)
# 添加一列，用于指定食物的来源
meat_to_animal = {
    'bacon': 'pig',
    'pulled pork': 'pig',
    'pastrami': 'cow',
    'corned beef': 'cow',
    'honey ham': 'pig',
    'nova lox': 'salmon'
}
# 使用map()
lowercased = data["food"].str.lower()
data["animal"] = lowercased.map(meat_to_animal)
# 使用方法
# lambda 函数就是将当前值作为参数进行计算 然后将返回值回写到当前值
data["animal2"] = data["food"].map(lambda x: meat_to_animal[x.lower()])
# Pandas 中map, applymap and apply的区别
# apply() 当想让方程作用在一维的向量上时，可以使用apply来完成但是因为大多数的列表统计方程
#     (比如 sum 和 mean)是DataFrame的函数，所以apply很多时候不是必须的
# applymap() 如果想让方程作用于DataFrame中的每一个元素，可以使用applymap().
# map() 只要是作用将函数作用于一个Series的每一个元素
# 总的来说就是apply()是一种让函数作用于列或者行操作，applymap()是一种让函数作用于DataFrame每一个元素的操作，
# 而map是一种让函数作用于Series每一个元素的操作

# 替换值
data = pd.Series([1., -999., 2., -999., -1000., 3.])
print(data)
# 将-999替换为空值
data.replace(-999, np.NaN)
# 将-999，-1000都替换为空值
data.replace([-999, -1000], np.nan)
data.replace({-999: np.nan, -1000: 0})

# 离散化和分箱操作
# 年龄数据
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# 分箱的边界
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins, right=False)
# Categorical对象
print(cats)
# 获取分箱编码
print(cats.codes)
print(cats.categories)
print(cats.get_values)
pd.value_counts(cats)
# 带标签的分箱
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
cats = pd.cut(ages, bins, labels=group_names)
cats.get_values()
print(cats.codes)

# 哑变量操作
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                   'data1': range(6)})
print(df)
pd.get_dummies(df, "key")
pd.get_dummies(df["key"])

# 向量化字符串操作
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
print(data)

data.str.contains("gmail")
split_df = data.str.split('@', expand=True)
print(split_df)
split_df[0].str.cat(split_df[1], sep='@')
# ------------------------------------------------------------------------------------------------------------
# 数据合并及分组
# ------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------
# 2.1. 数据合并
# ------------------------------------------------------
staff_df = pd.DataFrame([{'姓名': '张三', '部门': '研发部'},
                         {'姓名': '李四', '部门': '财务部'},
                         {'姓名': '赵六', '部门': '市场部'}])
student_df = pd.DataFrame([{'姓名': '张三', '专业': '计算机'},
                           {'姓名': '李四', '专业': '会计'},
                           {'姓名': '王五', '专业': '市场营销'}])
pd.merge(staff_df, student_df, how="outer", on="姓名")
pd.merge(staff_df, student_df, how="inner", on="姓名")
pd.merge(staff_df, student_df, how='left', on='姓名')
pd.merge(staff_df, student_df, how='right', on='姓名')
# 也可以按索引进行合并
staff_df.set_index("姓名", inplace=True)
student_df.set_index("姓名", inplace=True)
pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
# 当数据中的列名不同时，使用left_on，right_on
staff_df.reset_index(inplace=True)
student_df.reset_index(inplace=True)
# 也可指定后缀名称
staff_df.rename(columns={'姓名': '员工姓名'}, inplace=True)
student_df.rename(columns={'姓名': '学生姓名'}, inplace=True)
pd.merge(staff_df, student_df, how="outer", left_on="员工姓名", right_on="学生姓名", suffixes=['(公司)', '(家乡)'])
pd.merge(staff_df, student_df, how='left',  left_on='员工姓名', right_on='学生姓名', suffixes=('(公司)', '(家乡)'))
# apply使用
staff_df['员工姓名'].apply(lambda x: x[2])
staff_df["员工姓名"].apply(lambda x: x[2:])
# 结果合并
staff_df.loc[:,"姓"] = staff_df['员工姓名'].apply(lambda x: x[2])
# ------------------------------------------------------
#  2. 数据分组
# ------------------------------------------------------
report_data = pd.read_csv('./2015.csv')
report_data.head()

#groupby()
report_data.columns.value
grouped = report_data.groupby(['Region'])
grouped["Happiness Score"].mean()
grouped.size()
# 迭代groupby对象
for group, frame in grouped:
    mean_score = frame['Happiness Score'].mean()
    max_score = frame['Happiness Score'].max()
    min_score = frame['Happiness Score'].min()
    print('{}地区的平均幸福指数：{}，最高幸福指数：{}，最低幸福指数{}'.format(group, mean_score, max_score, min_score))
# 自定义函数进行分组
# 按照幸福指数排名进行划分，1-10, 10-20, >20
# 如果自定义函数，操作针对的是index
report_data2 = report_data.set_index('Happiness Rank')
def get_rank_group(rank):
    rank_group = ''
    if rank <= 10:
        rank_group = '0 -- 10'
    elif rank <= 20:
        rank_group = '10 -- 20'
    else:
        rank_group = '> 20'
    return rank_group
grouped = report_data2.groupby(get_rank_group)
for group, frame in grouped:
    print('{}分组的数据个数：{}'.format(group, len(frame)))
# 实际项目中，通常可以先人为构造出一个分组列，然后再进行groupby
# 按照score的整数部分进行分组
# 按照幸福指数排名进行划分，1-10, 10-20, >20
# 如果自定义函数，操作针对的是index
report_data["group"] = report_data["Happiness Score"].apply(lambda x: int(x))
grouped = report_data.groupby("'score group")
for group, frame in grouped:
    print("幸福指数整数部分为{}的分组数据个数：{}".format(group,len(frame)))
# agg是一个很方便的函数，它能针对分组后的列数据进行丰富多彩的计算。
# agg除了系统自带的几个函数，它也支持自定义函数。
grouped.agg({'Happiness Score': np.mean, 'Happiness Rank': np.max})
# def func():
#     return {'Happiness Score': np.mean, 'Happiness Rank': np.max}
# grouped['Happiness Score'].agg(func)
# ------------------------------------------------------------------------------------------------------------
# 透视表
# ------------------------------------------------------------------------------------------------------------
cars_df = pd.read_csv('cars.csv')
cars_df.head()
# 我们想要比较不同年份的不同厂商的车，在电池方面的不同
cars_df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)
# 我们想要比较不同年份的不同厂商的车，在电池方面的不同
cars_df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)
# 我们想要比较不同年份的不同厂商的车，在电池方面的不同
# 可以使用多个聚合函数
cars_df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean, np.min])
# 我们想要比较不同年份的不同厂商的车，在电池方面的不同
# 可以使用多个聚合函数
cars_df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean, np.min], margins=True)








