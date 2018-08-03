#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Index
# ----------------------------------------
# ks_ar               计算AR和KS



import numpy as np
import pandas as pd


def ks_ar(df, score, target):
    """
    计算AR和KS
    :param df:
    :param score:
    :param target:
    :return:
    """
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    df_all = pd.DataFrame({'total': total, 'bad': bad})
    df_all['good'] = df_all['total'] - df_all['bad']
    df_all[score] = df_all.index
    df_all = df_all.sort_values(by=score, ascending=False)
    df_all.index = range(len(df_all))
    df_all['badCumRate'] = df_all['bad'].cumsum() / df_all['bad'].sum()
    df_all['goodCumRate'] = df_all['good'].cumsum() / df_all['good'].sum()
    df_all['totalPcnt'] = df_all['total'] / df_all['total'].sum()
    arList = [0.5 * df_all.loc[0, 'badCumRate'] * df_all.loc[0, 'totalPcnt']]
    for j in range(1, len(df_all)):
        ar0 = 0.5 * sum(df_all.loc[j - 1:j, 'badCumRate']) * df_all.loc[j, 'totalPcnt']
        arList.append(ar0)
    arIndex = (2 * sum(arList) - 1) / (df_all['good'].sum() * 1.0 / df_all['total'].sum())
    KS = df_all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return {'AR': arIndex, 'KS': max(KS)}














