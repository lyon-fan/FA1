#! /usr/bin/env python
# -*- coding:utf-8 -*-


# Index
# ----------------------------------------
# create_path       创建文件夹
# list2txt          将list保存到txt中
# txt2list          从Txt中读取List
# create_frame      生成frame结构，便于查看
# clear_index_html  恢复index.html初始状态
# add_index_html    将索引写入index.html文件

import os


def create_frame(p_path, p_file_name):
    """
    生成frame结构，便于查看
    :param p_path: frame文件保存路径
    :param p_file_name: 属性名 函数将生成main_[p_file_name].html、index_[p_file_name].html、content_[p_file_name].html文件
    :return:
    """
    # p_path = r'D:\PythonWorkSpace\Python-Repository\Lesson\FA1\Result\Explore\1\test'
    # p_file_name = "1111"
    main_file_name = p_path+"\\main_"+p_file_name+".html"
    index_file_name = p_path+"\\index_"+p_file_name+".html"
    content_file_name = p_path+"\\content_"+p_file_name+".html"

    with open(main_file_name, 'w') as f:
        f.write("<html>\n")
        f.write("<frameset cols = '20%,*'>  \n")
        f.write("<frame name=index' src='index_"+p_file_name+".html' />  \n")
        f.write("<frame name='content' src='content_"+p_file_name+".html' />  \n")
        f.write("</frameset>  \n")
        f.write("</html>  \n")

    with open(index_file_name, 'w') as f:
        f.write("<a href='content_"+p_file_name+".html' target='content'>索引</a></br>\n")

    with open(content_file_name, 'w') as f:
        f.write("请选择属性进行查看</br>\n")

    print("function create_frame finished!...................")


def clear_index_html(p_path, p_file_name):
    """
    恢复index.html初始状态
    :param p_path: 文件路径
    :param p_file_name: 文件名称
    :return:
    """
    index_file_name = p_path + "\\index_" + p_file_name + ".html"
    with open(index_file_name, 'w') as f:
        f.write("<a href='content_"+p_file_name+".html' target='content'>索引</a></br>\n")


def add_index_html(p_path, p_file_name, p_var_name):
    """
    将索引写入index.html文件
    :param p_path:
    :param p_file_name:
    :param p_var_name:
    :return:
    """
    index_file_name = p_path + "\\index_" + p_file_name + ".html"
    with open(index_file_name, 'a+') as f:
        f.write("<a href='"+p_var_name+".png' target='content'>"+p_var_name+"</a></br>\n")


def list2txt(p_path, p_file, p_list):
    """
    将list保存到txt中
    :param p_path: 路径
    :param p_file: 文件名
    :param p_list: 要写入文件的list
    :return:
    """
    p_file = open(p_path + '\\' + p_file, 'w')
    for var in p_list:
        p_file.write(var)
        p_file.write('\n')
    p_file.close()
    print("function list2txt finished!...................")


def txt2list(p_file):
    """
    从Txt中读取List
    :param p_file: Txt文件
    :return: Txt文件中保存的List
    """
    a = open(p_file)
    lines = a.readlines()
    lists = []  # 直接用一个数组存起来就好了
    for line in lines:
        line = line.strip('\n')
        lists.append(line)
    print("function txt2list finished!...................")
    return lists


def create_path(p_path):
    """
    创建文件夹
    :param p_path:文件夹路径 
    :return: 
    """
    # 去除首位空格
    path = p_path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    is_exists = os.path.exists(path)

    # 判断结果
    if not is_exists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        print("function create_path finished!...................")
        # return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        print("function create_path finished!...................")
        # return False
