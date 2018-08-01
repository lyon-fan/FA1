#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import datetime
import collections
import pickle
from itertools import combinations
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import chinaPnr.utility.explore as u_explore
import chinaPnr.utility.modify as u_modify
import chinaPnr.utility.others as u_others
import chinaPnr.utility.sample as u_sample
import chinaPnr.utility.model as u_model
import chinaPnr.utility.assess as u_assess
# import io
# import sys
import numbers
# import numpy as np
# from matplotlib import pyplot


if __name__ == "__main__":
    # ##########################################################
    # #################0、原始参数设置            ##############
    # ##########################################################
    """
    路径设置
    """
    # 根目录
    path_root = os.getcwd()
    # 探索结果路径
    path_explore_result = path_root+"\\results\\explore"
    u_others.create_path(path_explore_result)
    path_detail = path_explore_result + "\\detail"
    u_others.create_path(path_detail)
    # CSV路径
    path_csv = path_root+"\\CSV"
    u_others.create_path(path_csv)
    # pkl路径
    path_pkl = path_root+"\\pkl"
    u_others.create_path(path_pkl)
    """
    公用字段设置
    """
    # ID的字段名
    col_id = "Idx"
    # 目标字段名
    col_target = "target"
    # 排除字段名
    drop_var = ["ListingInfo"]
    # #######################################################
    # ##############1、时间窗口设置            ##############
    # #######################################################
    """
    时间窗口
    """
    time_windows_data = pd.read_csv(path_csv+"\\"+"timeWindows.csv", header=0, encoding="gbk")
    time_windows = u_explore.time_window_selection(p_df=time_windows_data, p_days_col="ListingGap",
                                                   p_time_windows=range(30, 361, 30),
                                                   p_save_file=path_explore_result+"\\timeWindows.png")
    # 得到类别型和数字型变量名列表并保存
    """
    数据读取
    """
    allData = pd.read_csv(path_csv+"\\"+"allData_0.csv", header=0, encoding="gbk")
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=allData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    u_others.list2txt(path_explore_result, "var_string_list.csv", string_var_list)
    u_others.list2txt(path_explore_result, "var_number_list.csv", number_var_list)
    u_others.list2txt(path_explore_result, "all_var_list.csv", all_var_list)
    # -------------------------------------------------------------------------------------------------
    # todo 手动调字符串类型 连续型
    # todo 如果重新跑数据 或者调整字段则 用txt2list()重新加载即可
    string_var_list = u_others.txt2list(path_explore_result+"\\var_string_list.csv")
    number_var_list = u_others.txt2list(path_explore_result+"\\var_number_list.csv")
    all_var_list = u_others.txt2list(path_explore_result+"\\all_var_list.csv")
    # #########################################################
    # ################2、Sample                  ###############
    # #########################################################

    # #########################################################
    # ################3、Explore                 ###############
    # #########################################################
    """
    探索数据分布
    """
    u_explore.num_var_perf(p_df=allData, p_var_list=number_var_list, p_target_var=col_target,
                           p_path=path_explore_result)
    u_explore.str_var_pref(p_df=allData, p_var_list=string_var_list, p_target_var=col_target,
                           p_path=path_explore_result)
    # ##########################################################
    # #################4、Modify                  ###############
    # ##########################################################
    """
    处理异常值:去掉取值完全一样的数据
    """
    for col in all_var_list:
        if len(set(allData[col])) == 1:
            print("delete {} from the dataset because it is a constant".format(col))
            del allData[col]
            all_var_list.remove(col)
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=allData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    """
    处理异常值:去掉缺失值超过阈值的变量 连续变量0.3 字符变量0.5
    """
    u_modify.drop_num_missing_over_pcnt(p_df=allData,  p_num_var_list=number_var_list, p_threshould=0.3)
    u_modify.drop_str_missing_over_pcnt(p_df=allData,  p_str_var_list=string_var_list, p_threshould=0.5)

    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=allData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    u_others.list2txt(path_explore_result, "var_string_list.csv", string_var_list)
    u_others.list2txt(path_explore_result, "var_number_list.csv", number_var_list)
    u_others.list2txt(path_explore_result, "all_var_list.csv", all_var_list)

    u_explore.missing_categorial(allData, string_var_list, path_explore_result+"\\"+"missing_categorial.csv")
    u_explore.missing_continuous(allData, number_var_list, path_explore_result+"\\"+"missing_num.csv")

    allData_bk = allData.copy()
    """
    天花板地板法：处理过大过小值
    """
    # todo 去掉异常值
    """
    填缺：缺失值填补
    """
    u_modify.makeup_num_miss(allData, number_var_list, "PERC50")
    u_explore.missing_continuous(allData, number_var_list, path_explore_result+"\\"+"missing_num02.csv")
    u_modify.makeup_str_miss(allData, string_var_list, "MODE")
    u_explore.missing_categorial(allData, string_var_list, path_explore_result+"\\"+"missing_categorial02.csv")

    allData.to_csv(path_csv+"\\"+"allData_makeup_missing.csv", header=True, encoding="gbk", columns=allData.columns,
                   index=False)

    """
    变量分组 计算WOE IV
    """
    trainData = pd.read_csv(path_csv+"\\"+"allData_1_bak.csv", header=0, encoding="gbk")
    string_var_list, number_var_list, all_var_list = u_explore.get_list_for_number_str_col(p_df=trainData,
                                                                                           p_col_id=col_id,
                                                                                           p_col_target=col_target,
                                                                                           p_drop_col=drop_var)
    string_var_list = ["UserInfo_1",
                       "UserInfo_3",
                       "WeblogInfo_2",
                       "WeblogInfo_13",
                       "UserInfo_5",
                       "UserInfo_6",
                       "UserInfo_7",
                       "UserInfo_9",
                       "UserInfo_10",
                       "UserInfo_14",
                       "UserInfo_15",
                       "UserInfo_16",
                       "UserInfo_17",
                       "UserInfo_19",
                       "UserInfo_21",
                       "UserInfo_22",
                       "UserInfo_23",
                       "UserInfo_24",
                       "Education_Info1",
                       "Education_Info2",
                       "Education_Info3",
                       "Education_Info4",
                       "Education_Info5",
                       "Education_Info6",
                       "Education_Info7",
                       "Education_Info8",
                       "WeblogInfo_19",
                       "WeblogInfo_20",
                       "WeblogInfo_21",
                       "WeblogInfo_23",
                       "WeblogInfo_25",
                       "WeblogInfo_26",
                       "WeblogInfo_28",
                       "WeblogInfo_29",
                       "WeblogInfo_30",
                       "WeblogInfo_31",
                       "WeblogInfo_32", "WeblogInfo_34", "WeblogInfo_35", "WeblogInfo_37",
                       "WeblogInfo_38", "WeblogInfo_39", "WeblogInfo_40", "WeblogInfo_41", "WeblogInfo_43",
                       "WeblogInfo_44", "WeblogInfo_45", "WeblogInfo_46", "WeblogInfo_47", "WeblogInfo_48",
                       "WeblogInfo_49", "WeblogInfo_50", "WeblogInfo_51", "WeblogInfo_52", "WeblogInfo_53",
                       "WeblogInfo_54", "WeblogInfo_55", "WeblogInfo_56", "WeblogInfo_57",
                       "WeblogInfo_58", "SocialNetwork_1", "SocialNetwork_2", "SocialNetwork_7",
                       "SocialNetwork_11", "SocialNetwork_12", "SocialNetwork_13",
                       "SocialNetwork_14", "SocialNetwork_15", "SocialNetwork_16", "SocialNetwork_17", "city_match"
                       ]

    number_var_list = ["ListingGap",
                       "WeblogInfo_4",
                       "WeblogInfo_5",
                       "WeblogInfo_6",
                       "WeblogInfo_7",
                       "WeblogInfo_8",
                       "WeblogInfo_9",
                       "WeblogInfo_11",
                       "WeblogInfo_12",
                       "WeblogInfo_14",
                       "WeblogInfo_15",
                       "WeblogInfo_16",
                       "WeblogInfo_17",
                       "WeblogInfo_18",
                       "UserInfo_12",
                       "UserInfo_18",
                       "WeblogInfo_24",
                       "WeblogInfo_27",
                       "WeblogInfo_33",
                       "WeblogInfo_36",
                       "WeblogInfo_42",
                       "ThirdParty_Info_Period1_1",
                       "ThirdParty_Info_Period1_2",
                       "ThirdParty_Info_Period1_3",
                       "ThirdParty_Info_Period1_4",
                       "ThirdParty_Info_Period1_5",
                       "ThirdParty_Info_Period1_6",
                       "ThirdParty_Info_Period1_7",
                       "ThirdParty_Info_Period1_8",
                       "ThirdParty_Info_Period1_9",
                       "ThirdParty_Info_Period1_10",
                       "ThirdParty_Info_Period1_11",
                       "ThirdParty_Info_Period1_12",
                       "ThirdParty_Info_Period1_13",
                       "ThirdParty_Info_Period1_14",
                       "ThirdParty_Info_Period1_15",
                       "ThirdParty_Info_Period1_16",
                       "ThirdParty_Info_Period1_17",
                       "ThirdParty_Info_Period2_1",
                       "ThirdParty_Info_Period2_2",
                       "ThirdParty_Info_Period2_3",
                       "ThirdParty_Info_Period2_4",
                       "ThirdParty_Info_Period2_5",
                       "ThirdParty_Info_Period2_6",
                       "ThirdParty_Info_Period2_7",
                       "ThirdParty_Info_Period2_8",
                       "ThirdParty_Info_Period2_9",
                       "ThirdParty_Info_Period2_10",
                       "ThirdParty_Info_Period2_11",
                       "ThirdParty_Info_Period2_12",
                       "ThirdParty_Info_Period2_13",
                       "ThirdParty_Info_Period2_14",
                       "ThirdParty_Info_Period2_15",
                       "ThirdParty_Info_Period2_16",
                       "ThirdParty_Info_Period2_17",
                       "ThirdParty_Info_Period3_1",
                       "ThirdParty_Info_Period3_2",
                       "ThirdParty_Info_Period3_3",
                       "ThirdParty_Info_Period3_4",
                       "ThirdParty_Info_Period3_5",
                       "ThirdParty_Info_Period3_6",
                       "ThirdParty_Info_Period3_7",
                       "ThirdParty_Info_Period3_8",
                       "ThirdParty_Info_Period3_9",
                       "ThirdParty_Info_Period3_10",
                       "ThirdParty_Info_Period3_11",
                       "ThirdParty_Info_Period3_12",
                       "ThirdParty_Info_Period3_13",
                       "ThirdParty_Info_Period3_14",
                       "ThirdParty_Info_Period3_15",
                       "ThirdParty_Info_Period3_16",
                       "ThirdParty_Info_Period3_17",
                       "ThirdParty_Info_Period4_1",
                       "ThirdParty_Info_Period4_2",
                       "ThirdParty_Info_Period4_3",
                       "ThirdParty_Info_Period4_4",
                       "ThirdParty_Info_Period4_5",
                       "ThirdParty_Info_Period4_6",
                       "ThirdParty_Info_Period4_7",
                       "ThirdParty_Info_Period4_8",
                       "ThirdParty_Info_Period4_9",
                       "ThirdParty_Info_Period4_10",
                       "ThirdParty_Info_Period4_11",
                       "ThirdParty_Info_Period4_12",
                       "ThirdParty_Info_Period4_13",
                       "ThirdParty_Info_Period4_14",
                       "ThirdParty_Info_Period4_15",
                       "ThirdParty_Info_Period4_16",
                       "ThirdParty_Info_Period4_17",
                       "ThirdParty_Info_Period5_1",
                       "ThirdParty_Info_Period5_2",
                       "ThirdParty_Info_Period5_3",
                       "ThirdParty_Info_Period5_4",
                       "ThirdParty_Info_Period5_5",
                       "ThirdParty_Info_Period5_6",
                       "ThirdParty_Info_Period5_7",
                       "ThirdParty_Info_Period5_8",
                       "ThirdParty_Info_Period5_9",
                       "ThirdParty_Info_Period5_10",
                       "ThirdParty_Info_Period5_11",
                       "ThirdParty_Info_Period5_12",
                       "ThirdParty_Info_Period5_13",
                       "ThirdParty_Info_Period5_14",
                       "ThirdParty_Info_Period5_15",
                       "ThirdParty_Info_Period5_16",
                       "ThirdParty_Info_Period5_17",
                       "ThirdParty_Info_Period6_1",
                       "ThirdParty_Info_Period6_2",
                       "ThirdParty_Info_Period6_3",
                       "ThirdParty_Info_Period6_4",
                       "ThirdParty_Info_Period6_5",
                       "ThirdParty_Info_Period6_6",
                       "ThirdParty_Info_Period6_7",
                       "ThirdParty_Info_Period6_8",
                       "ThirdParty_Info_Period6_9",
                       "ThirdParty_Info_Period6_10",
                       "ThirdParty_Info_Period6_11",
                       "ThirdParty_Info_Period6_12",
                       "ThirdParty_Info_Period6_13",
                       "ThirdParty_Info_Period6_14",
                       "ThirdParty_Info_Period6_15",
                       "ThirdParty_Info_Period6_16",
                       "ThirdParty_Info_Period6_17",
                       "ThirdParty_Info_Period7_1",
                       "ThirdParty_Info_Period7_2",
                       "ThirdParty_Info_Period7_3",
                       "ThirdParty_Info_Period7_4",
                       "ThirdParty_Info_Period7_5",
                       "ThirdParty_Info_Period7_6",
                       "ThirdParty_Info_Period7_7",
                       "ThirdParty_Info_Period7_8",
                       "ThirdParty_Info_Period7_9",
                       "ThirdParty_Info_Period7_10",
                       "ThirdParty_Info_Period7_11",
                       "ThirdParty_Info_Period7_12",
                       "ThirdParty_Info_Period7_13",
                       "ThirdParty_Info_Period7_14",
                       "ThirdParty_Info_Period7_15",
                       "ThirdParty_Info_Period7_16",
                       "ThirdParty_Info_Period7_17",
                       "SocialNetwork_3",
                       "SocialNetwork_4",
                       "SocialNetwork_5",
                       "SocialNetwork_6",
                       "SocialNetwork_8",
                       "SocialNetwork_9",
                       "SocialNetwork_10",
                       "UserupdateInfo_7_freq",
                       "UserupdateInfo_7_unique",
                       "UserupdateInfo_7_avg_count",
                       "UserupdateInfo_7_IDNUMBER",
                       "UserupdateInfo_7_HASBUYCAR",
                       "UserupdateInfo_7_MARRIAGESTATUSID",
                       "UserupdateInfo_7_PHONE",
                       "UserupdateInfo_30_freq",
                       "UserupdateInfo_30_unique",
                       "UserupdateInfo_30_avg_count",
                       "UserupdateInfo_30_IDNUMBER",
                       "UserupdateInfo_30_HASBUYCAR",
                       "UserupdateInfo_30_MARRIAGESTATUSID",
                       "UserupdateInfo_30_PHONE",
                       "UserupdateInfo_60_freq",
                       "UserupdateInfo_60_unique",
                       "UserupdateInfo_60_avg_count",
                       "UserupdateInfo_60_IDNUMBER",
                       "UserupdateInfo_60_HASBUYCAR",
                       "UserupdateInfo_60_MARRIAGESTATUSID",
                       "UserupdateInfo_60_PHONE",
                       "UserupdateInfo_90_freq",
                       "UserupdateInfo_90_unique",
                       "UserupdateInfo_90_avg_count",
                       "UserupdateInfo_90_IDNUMBER",
                       "UserupdateInfo_90_HASBUYCAR",
                       "UserupdateInfo_90_MARRIAGESTATUSID",
                       "UserupdateInfo_90_PHONE",
                       "UserupdateInfo_120_freq",
                       "UserupdateInfo_120_unique",
                       "UserupdateInfo_120_avg_count",
                       "UserupdateInfo_120_IDNUMBER",
                       "UserupdateInfo_120_HASBUYCAR",
                       "UserupdateInfo_120_MARRIAGESTATUSID",
                       "UserupdateInfo_120_PHONE",
                       "UserupdateInfo_150_freq",
                       "UserupdateInfo_150_unique",
                       "UserupdateInfo_150_avg_count",
                       "UserupdateInfo_150_IDNUMBER",
                       "UserupdateInfo_150_HASBUYCAR",
                       "UserupdateInfo_150_MARRIAGESTATUSID",
                       "UserupdateInfo_150_PHONE",
                       "UserupdateInfo_180_freq",
                       "UserupdateInfo_180_unique",
                       "UserupdateInfo_180_avg_count",
                       "UserupdateInfo_180_IDNUMBER",
                       "UserupdateInfo_180_HASBUYCAR",
                       "UserupdateInfo_180_MARRIAGESTATUSID",
                       "UserupdateInfo_180_PHONE",
                       "LogInfo1_7_count",
                       "LogInfo1_7_unique",
                       "LogInfo1_7_avg_count",
                       "LogInfo2_7_count",
                       "LogInfo2_7_unique",
                       "LogInfo2_7_avg_count",
                       "LogInfo1_30_count",
                       "LogInfo1_30_unique",
                       "LogInfo1_30_avg_count",
                       "LogInfo2_30_count",
                       "LogInfo2_30_unique",
                       "LogInfo2_30_avg_count",
                       "LogInfo1_60_count",
                       "LogInfo1_60_unique",
                       "LogInfo1_60_avg_count",
                       "LogInfo2_60_count",
                       "LogInfo2_60_unique",
                       "LogInfo2_60_avg_count",
                       "LogInfo1_90_count",
                       "LogInfo1_90_unique",
                       "LogInfo1_90_avg_count",
                       "LogInfo2_90_count",
                       "LogInfo2_90_unique",
                       "LogInfo2_90_avg_count",
                       "LogInfo1_120_count",
                       "LogInfo1_120_unique",
                       "LogInfo1_120_avg_count",
                       "LogInfo2_120_count",
                       "LogInfo2_120_unique",
                       "LogInfo2_120_avg_count",
                       "LogInfo1_150_count",
                       "LogInfo1_150_unique",
                       "LogInfo1_150_avg_count",
                       "LogInfo2_150_count",
                       "LogInfo2_150_unique",
                       "LogInfo2_150_avg_count",
                       "LogInfo1_180_count",
                       "LogInfo1_180_unique",
                       "LogInfo1_180_avg_count",
                       "LogInfo2_180_count",
                       "LogInfo2_180_unique",
                       "LogInfo2_180_avg_count"]
    all_var_list = string_var_list + number_var_list

    for col in string_var_list:
        if col not in ["UserInfo_7", "UserInfo_9", "UserInfo_19", "UserInfo_22", "UserInfo_23", "UserInfo_24",
                       "Education_Info3", "Education_Info7", "Education_Info8"]:
            trainData[col] = trainData[col].map(lambda x: str(x).upper())

    deleted_var_list = []
    encoded_var_dict = {}
    merged_var_dict = {}
    var_iv_dict = {}
    var_woe_dict = {}
    u_modify.woe_iv_for_string(p_df=trainData,
                               p_str_var_list=string_var_list, p_num_var_list=number_var_list,
                               p_target=col_target,
                               p_deleted_var_list=deleted_var_list, p_encoded_var_dict=encoded_var_dict,
                               p_merged_var_dict=merged_var_dict, p_var_iv_dict=var_iv_dict,
                               p_var_woe_dict=var_woe_dict)
    var_cutoff_dict = {}
    number_var_list.remove("ListingGap")
    number_var_list.remove("UserInfo_12")
    # number_var_list=["WeblogInfo_13_encoding"]
    u_modify.woe_iv_for_num(p_df=trainData, p_str_num_list=number_var_list, p_target=col_target,
                            p_deleted_var_list=deleted_var_list, p_var_iv_dict=var_iv_dict,
                            p_var_woe_dict=var_woe_dict, p_var_cutoff_dict=var_cutoff_dict)
    """
    保存结果
    """
    trainData.to_csv(path_csv+"\\"+"allData_2.csv", header=True, encoding="gbk", columns=trainData.columns,
                     index=False)
    var_woe_file = open(path_pkl+"\\"+"var_woe_dict.pkl", "wb")
    pickle.dump(var_woe_dict, var_woe_file)
    var_woe_file.close()

    var_iv_file = open(path_pkl+"\\"+"var_iv_dict.pkl", "wb")
    pickle.dump(var_iv_dict, var_iv_file)
    var_iv_file.close()

    var_cutoff_file = open(path_pkl+"\\"+"var_cutoff_dict.pkl", "wb")
    pickle.dump(var_cutoff_dict, var_cutoff_file)
    var_cutoff_file.close()

    var_merged_file = open(path_pkl+"\\"+"merged_var_dict.pkl", "wb")
    pickle.dump(merged_var_dict, var_merged_file)
    var_merged_file.close()
    # #########################################################
    # ################5、Model                   ###############
    # #########################################################
    """
    选择IV>0.02的变量 并标注WOE
    """
    trainData = pd.read_csv(path_csv+"\\"+"allData_2.csv", header=0, encoding="gbk")

    with open(path_pkl+"\\"+"var_woe_dict.pkl", "rb") as f:
        var_woe_dict = pickle.load(f, encoding="gbk")

    with open(path_pkl+"\\"+"var_iv_dict.pkl", "rb") as f:
        var_iv_dict = pickle.load(f, encoding="gbk")

    with open(path_pkl+"\\"+"var_cutoff_dict.pkl", "rb") as f:
        var_cutoff_dict = pickle.load(f, encoding="gbk")

    with open(path_pkl+"\\"+"merged_var_dict.pkl", "rb") as f:
        merged_var_dict = pickle.load(f, encoding="gbk")

    # 将一些看起来像数值变量实际上是类别变量的字段转换成字符
    num2str = ["SocialNetwork_13", "SocialNetwork_12", "UserInfo_6", "UserInfo_5", "UserInfo_10",
               "UserInfo_17", "city_match"]
    for col in num2str:
        trainData[col] = trainData[col].map(lambda x: str(x))

    # 将所有变量WOE编码
    for var in var_woe_dict:
        # var = "WeblogInfo_21"
        print(var)
        var2 = str(var)+"_WOE"
        if var in var_cutoff_dict.keys():
            cutOffPoints = var_cutoff_dict[var]
            # special_attribute = []
            # if - 1 in cutOffPoints:
            #     special_attribute = [-1]
            binValue = trainData[var].map(lambda x: u_modify.assign_bin(x, cutOffPoints))
            trainData[var2] = binValue.map(lambda x: var_woe_dict[var][x])
        else:
            trainData[var2] = trainData[var].map(lambda x: var_woe_dict[var][x])
    trainData.to_csv("allData_3.csv", header=True, encoding="gbk", columns=trainData.columns, index=False)

    # 选择IV高于阈值的变量
    trainData = pd.read_csv(path_csv+"\\"+"allData_3.csv", header=0, encoding="gbk")
    all_iv = list(var_iv_dict.values())
    all_iv = sorted(all_iv, reverse=True)
    plt.bar(x=range(len(all_iv)), height=all_iv)
    plt.show()
    iv_threshould = 0.02
    var_IV_sorted_1 = [k for k, v in var_iv_dict.items() if v > iv_threshould]

    # todo 针对变量IV值 进行变量挑选 不然两两线性相关会计算量可能很大
    """
    检查WOE编码后的变量的两两线性相关性
    """
    var_IV_selected = {k: var_iv_dict[k] for k in var_IV_sorted_1}
    compare = list(combinations(var_IV_selected, 2))
    removed_var = []
    roh_thresould = 0.8
    for pair in compare:
        (x1, x2) = pair
        roh = np.corrcoef([trainData[str(x1)+"_WOE"], trainData[str(x2)+"_WOE"]])[0, 1]
        if abs(roh) >= roh_thresould:
            if var_iv_dict[x1] > var_iv_dict[x2]:
                removed_var.append(x2)
            else:
                removed_var.append(x1)
    var_IV_sorted_2 = [i for i in var_IV_selected if i not in removed_var]

    """
    检查是否有变量与其他所有变量的VIF > 10
    """
    for i in range(len(var_IV_sorted_2)):
        x0 = trainData[var_IV_sorted_2[i] + "_WOE"]
        x0 = np.array(x0)
        X_Col = [k + "_WOE" for k in var_IV_sorted_2 if k != var_IV_sorted_2[i]]
        X = trainData[X_Col]
        X = np.matrix(X)
        regr = LinearRegression()
        clr = regr.fit(X, x0)
        x_pred = clr.predict(X)
        R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
        vif = 1 / (1 - R2)
        if vif > 10:
            print("Warning: the vif for {0} is {1}".format(var_IV_sorted_2[i], vif))

    """
    建模1
    """
    var_WOE_list = [i+"_WOE" for i in var_IV_sorted_2]
    y = trainData[col_target]
    X = trainData[var_WOE_list]
    X["intercept"] = [1]*X.shape[0]

    LR = sm.Logit(y, X).fit()
    summary = LR.summary()
    pvals = LR.pvalues
    pvals = pvals.to_dict()

    # ##########################################################
    # #################6、Assess                  ###############
    # ##########################################################
















# # ##########################################################
# # #################原始数据处理              #################
# # ##########################################################
# # 根目录
# path_root = os.getcwd()
# # 路径
# path_explore_result = path_root+"\\Result\\Explore"
# u_others.create_path(path_explore_result)
# # ID的字段名
# col_id = "CUST_ID"
# # 目标字段名
# col_target = "CHURN_CUST_IND"
#
# # 合并数据
# data_bank = pd.read_csv(path_root + "\\bankChurn.csv")
# data_external = pd.read_csv(path_root + "\\ExternalData.csv")
# data_all = pd.merge(data_bank, data_external, on=col_id)
# data_all.head(5)
# # #########################################################
# # ###              数据探索                     #############
# # #########################################################
# # 得到类别型和数字型变量名列表并保存
# string_var_list, number_var_list = u_explore.get_list_for_number_str_col(p_df=data_all, p_col_id=col_id,
#                                                                          p_col_target=col_target)
# u_others.list2txt(path_explore_result, "var_string_list.txt", string_var_list)
# u_others.list2txt(path_explore_result, "var_number_list.txt", number_var_list)
# # data_all[string_var_list]
# # data_all[number_var_list]
# # todo 调用小程序手动调整
# # todo 如果重新跑数据 或者调整字段则 用txt2list()重新加载即可
# # string_var_list = txt2list(path_explore_result+"\\var_string_list.txt")
# # number_var_list = txt2list(path_explore_result+"\\var_number_list.txt")
#
# # 分别进行数字型变量和字符串变量的探索

#
# # 选择15个数字变量 看相关性
# # corr_cols = random.sample(number_var_list, 15)
# # sample_df = data_all[corr_cols]
# # scatter_matrix(sample_df, alpha=0.2, figsize=(14, 8), diagonal="kde")
# # plt.show()
#
# # 缺失值填充
# u_modify.makeup_num_miss(p_df=data_all, p_var_list=number_var_list, p_method="MEAN")
# u_modify.makeup_str_miss(p_df=data_all, p_str_var_list=string_var_list, p_method="MODE")
#
# # 浓度编码
# u_modify.density_encoder(data_all, string_var_list, col_target)
#
# # 卡方分箱
# cutoff_points = u_model.chi_merge_max_interval(data_all, "day", col_target, 5)
# var_cutoff = {}
# var_cutoff["day"] = cutoff_points
# data_all["day"] = data_all["day"].map(lambda x: u_model.assign_bin(x, cutoff_points))