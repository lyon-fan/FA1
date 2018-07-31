#! /usr/bin/env python
# -*- coding:utf-8 -*-


# Index
# ----------------------------------------
# get_list_for_number_str_col 将dataframe中的字段名称分为字符型、数值型两个list返回
# num_var_perf    探索数值型变量的分布
# str_var_pref    探索字符型变量的分布
# time_window_selection   根据给定的时间窗口 进行累计计数 保存图像 并返回结果
# missing_categorial_for_1 为某一个类别型变量统计缺失值
# missing_continuous_for_1 为某一个数值型变量统计缺失值 返回缺失率
# missing_categorial 为df中所有类别型变量统计缺失值
# missing_continuous 为df中所有连续型变量统计缺失值
# max_bin_pcnt 计算各个类别的占比 返回series

import numpy as np
import pandas as pd
from matplotlib import pyplot
import chinaPnr.utility.others as u_others


def get_list_for_number_str_col(p_df, p_col_id, p_col_target, p_drop_col=[] ):
    """
    将dataframe中的字段名称分为数值型、字符型、全部三个list返回
    这里只看数据类型 如果字典取值是数字 请在数据处理的时候 在最前面加上一个字母 推荐'A_'
    :param p_df: 数据集
    :param p_col_id: 主键字段名
    :param p_col_target: 目标字段名
    :return:str_var_list: 字符型变量列表；numberVarlist- 数值型变量列表
    """
    # p_df=allData
    # p_col_id=col_id
    # p_col_target=col_target
    # p_drop_col=drop_var

    name_of_col = list(p_df.columns)
    name_of_col.remove(p_col_target)
    name_of_col.remove(p_col_id)
    num_var_list = []
    str_var_list = []
    all_var_list = []

    str_var_list = name_of_col.copy()
    for var in name_of_col:
        # if (len(list(set(p_df[var]))) > 5) and (p_df[var].dtypes in (np.int, np.int64, np.uint, np.int32, np.float,
        #                                                             np.float64, np.float32, np.double)):
        if (p_df[var].dtypes in (np.int, np.int64, np.uint, np.int32, np.float,
                                                                         np.float64, np.float32, np.double)):
            str_var_list.remove(var)
            num_var_list.append(var)

    for var in p_drop_col:
        if var in str_var_list:
            str_var_list.remove(var)
        if var in num_var_list:
            num_var_list.remove(var)

    all_var_list.extend(str_var_list)
    all_var_list.extend(num_var_list)
    print("function get_list_for_number_str_col finished!...................")
    return str_var_list, num_var_list, all_var_list


def num_var_perf(p_df, p_var_list, p_target_var, p_path, p_truncation=False):
    """
    探索数值型变量的分布
    :param p_df: 数据集
    :param p_var_list:数值型变量名称列表 List类型
    :param p_target_var: 响应变量名称
    :param p_path: 保存图片的位置
    :param p_truncation: 是否对数据做95%盖帽处理 默认不盖帽
    :return:
    """
    frame_name = "num"

    u_others.create_frame(p_path, frame_name)

    for var in p_var_list:
        # 利用NaN != NaN的特性 将所有空值排除
        valid_df = p_df.loc[p_df[var] == p_df[var]][[var, p_target_var]]
        rec_perc = 100.0*valid_df.shape[0] / p_df.shape[0]
        rec_perc_fmt = "%.2f%%" % rec_perc
        desc = valid_df[var].describe()
        value_per50 = '%.2e' % desc['50%']
        value_std = '%.2e' % desc['std']
        value_mean = '%.2e' % desc['mean']
        value_max = '%.2e' % desc['max']
        value_min = '%.2e' % desc['min']
        # 样本权重
        bad_df = valid_df.loc[valid_df[p_target_var] == 1][var]
        good_df = valid_df.loc[valid_df[p_target_var] == 0][var]
        bad_weight = 100.0*np.ones_like(bad_df)/bad_df.size
        good_weight = 100.0*np.ones_like(good_df)/good_df.size
        # 是否用95分位数进行盖帽
        if p_truncation:
            per95 = np.percentile(valid_df[var], 95)
            bad_df = bad_df.map(lambda x: min(x, per95))
            good_df = good_df.map(lambda x: min(x, per95))
        # 画图
        fig, ax = pyplot.subplots()
        ax.hist(bad_df, weights=bad_weight, alpha=0.3, label='bad')
        ax.hist(good_df, weights=good_weight, alpha=0.3, label='good')
        title_text = var + '\n' \
                     + 'VlidePerc:' \
                     + rec_perc_fmt \
                     + ';Mean:' \
                     + value_mean \
                     + ';Per50:' + value_per50 \
                     + ';Std:' + value_std \
                     + ';\n' \
                     + 'Max:' + value_max \
                     + ';Min:'+value_min
        ax.set(title=title_text, ylabel='% of dataset in bin')
        ax.margins(0.05)
        ax.set_ylim(bottom=0)
        pyplot.legend(loc='upper right')
        fig_save_path = p_path + '\\' + str(var) + '.png'
        pyplot.savefig(fig_save_path)
        pyplot.close(1)
        u_others.add_index_html(p_path, frame_name, var)
        # pyplot.show()
    print("function num_var_perf finished!...................")


def str_var_pref(p_df, p_var_list, p_target_var, p_path):
    """
    探索字符型变量的分布
    :param p_df: 数据集
    :param p_var_list: 字符型型变量名称列表 List类型
    :param p_target_var: 响应变量名称
    :param p_path: 保存图片的位置
    :return:
    """

    frame_name = "str"
    u_others.create_frame(p_path, frame_name)

    for var in p_var_list:
        # 利用None != None的特性 将所有空值排除
        valid_df = p_df.loc[p_df[var] == p_df[var]][[var, p_target_var]]
        rec_perc = 100.0*valid_df.shape[0] / p_df.shape[0]
        rec_perc_fmt = "%.2f%%" % rec_perc
        dict_freq = {}
        dict_bad_rate = {}
        for v in set(valid_df[var]):
            v_df = valid_df.loc[valid_df[var] == v]
            # 每个类别数量占比
            dict_freq[v] = 1.0*v_df.shape[0] / p_df.shape[0]
            # 每个类别坏客户占比
            dict_bad_rate[v] = sum(v_df[p_target_var] * 1.0) / v_df[p_target_var].shape[0]

        if p_df.loc[p_df[var] != p_df[var]][p_target_var].shape[0] > 0:
            # 当前变量缺失率统计
            dict_freq['missValue'] = 1.0 - valid_df.shape[0] / p_df.shape[0]
            # 当前变量缺失率值中坏商户占比
            dict_bad_rate['missValue'] = \
                sum(p_df.loc[p_df[var] != p_df[var]][p_target_var]) \
                / p_df.loc[p_df[var] != p_df[var]][p_target_var].shape[0]
        desc_state = pd.DataFrame({'percent': dict_freq, 'bad rate': dict_bad_rate})
        # 画图
        fig = pyplot.figure()
        ax0 = fig.add_subplot(111)
        ax1 = ax0.twinx()
        pyplot.title('The percentage and bad rate for '+var+'\n valid rec_perc ='+rec_perc_fmt)
        desc_state.percent.plot(kind='bar', color='blue', ax=ax0, width=0.2, position=1)
        desc_state['bad rate'].plot(kind='line', color='red', ax=ax1)
        ax0.set_ylabel('percent(bar)')
        ax1.set_ylabel('bad rate(line)')
        fig_save_path = p_path + '\\' + str(var) + '.png'
        pyplot.savefig(fig_save_path)
        pyplot.close(1)
        u_others.add_index_html(p_path, frame_name, var)
    print("function str_var_pref finished!...................")


def time_window_selection(p_df, p_days_col, p_time_windows, p_save_file):
    """
    根据给定的时间窗口 进行累计计数 保存图像 并返回结果
    :param p_df: 数据集
    :param p_days_col: 天数所在的列
    :param p_time_windows: 时间窗口 为list 如[10,20,30]
    :param p_save_file: 保存文件全文件名
    :return:
    """
    total = p_df.shape[0]
    freq_pd = pd.DataFrame({"days": [0], "pcnt": [0.0]})
    for tw in p_time_windows:
        freq = sum(p_df[p_days_col].apply(lambda x: int(x <= tw)))
        freq_pd = freq_pd.append({"days": tw, "pcnt": freq * 100.0 / total}, ignore_index=True)
    # 画图
    fig = pyplot.figure()
    pyplot.grid()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(0, 100, 5))
    ax.set_ylabel("percent %")
    ax.plot(freq_pd["days"], freq_pd["pcnt"], "ro-")
    pyplot.savefig(p_save_file)
    pyplot.close(1)

    print("function time_window_selection finished!...................")
    return freq_pd


def missing_categorial_for_1(p_df, p_var):
    """
    为某一个类别型变量统计缺失值
    :param p_df:
    :param p_var:
    :return: 返回缺失率
    """
    missing_vals = p_df[p_var].map(lambda x: int(x != x))
    print("function missing_categorial_for_1:"+p_var+" finished!...................")
    return sum(missing_vals) * 1.0 / p_df.shape[0]


def missing_continuous_for_1(p_df, p_var):
    """
    为某一个数值型变量统计缺失值 返回缺失率
    :param p_df:
    :param p_var:
    :return:
    """
    missing_vals = p_df[p_var].map(lambda x: int(np.isnan(x)))
    print("function missing_continuous_for_1:"+p_var+"  finished!...................")
    return sum(missing_vals) * 1.0 / p_df.shape[0]


def missing_categorial(p_df, p_var_list, p_file=""):
    """
    为df中所有类别型变量统计缺失值
    :param p_df:
    :param p_var_list:  类别型变量的list
    :param p_file: 保存文件名称
    :return:
    """
    dict_mis = {}
    for v in p_var_list:
        dict_mis[v] = missing_categorial_for_1(p_df=p_df, p_var=v)

    if len(p_file.strip()) > 0:
        f = open(p_file, 'w')
        for key, value in dict_mis.items():
            f.write(key+","+str(value))
            f.write('\n')
        # f.write(str(dict_mis))
        # f.write('\n')
        f.close()

    print("function missing_categorial finished!...................")
    return dict_mis


def missing_continuous(p_df, p_var_list, p_file=""):
    """
    为df中所有连续型变量统计缺失值
    :param p_df:
    :param p_var_list: 连续型变量的list
    :param p_file: 保存文件名称
    :return:
    """
    dict_mis = {}
    for v in p_var_list:
        dict_mis[v] = missing_continuous_for_1(p_df=p_df, p_var=v)

    if len(p_file.strip()) > 0:
        f = open(p_file, 'w')
        for key, value in dict_mis.items():
            f.write(key+","+str(value))
            f.write('\n')
        # f.write(str(dict_mis))
        f.close()

    print("function missing_continuous finished!...................")
    return dict_mis


def max_bin_pcnt(p_df, col):
    """
    计算各个类别的占比 返回series
    :param p_df:
    :param col:
    :return: 返回series
    """
    n = p_df.shape[0]
    total = p_df.groupby([col])[col].count()
    pcnt = total*1.0/n
    pcnt.sort_values(inplace=True)
    print("function max_bin_pcnt finished!...................")
    return pcnt


