# -*- coding: utf-8 -*-
# @Time    : 2020/11/29 下午4:33
# @Author  : Chaos
# @FileName: commonFunc.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
from IPython import display
from matplotlib import pyplot as plt


def use_svg_display():
    # 生成散点图
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize