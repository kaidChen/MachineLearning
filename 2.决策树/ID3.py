import numpy as np
import pandas as pd
from collections import Counter


# information entropy
def Entropy(dataset, label_tag="好瓜"):
    ent, label = 0, dataset[label_tag]
    for k in np.unique(label):
        p = len(label[label == k]) / len(label)
        ent -= p * np.log2(p)
    return ent


# information gain
def Gain(dataset, attribute):
    w = 0
    for v in np.unique(dataset[attribute]):
        subset = dataset[dataset[attribute] == v]
        w += Entropy(subset) * len(subset) / len(dataset)
    return Entropy(dataset) - w


# TreeNode
class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    def __str__(self):
        if len(self.children) == 0:
            return str(self.attribute)
        children = ["%s: %s" % (child, self.children[child]) for child in self.children]
        children = str.join(", ", children)
        return "{%s {%s}}" % (self.attribute, children)


def TreeGenerate(dataset, attribute_set, attr_value_map, label_tag="好瓜"):
    if len(np.unique(dataset[label_tag])) == 1 or len(attribute_set) == 0:
        # 样本全部属于同一类别 或 当前属性集为空
        most = Counter(dataset[label_tag]).most_common()[0][0]
        return Node(most)
    # 根据信息增益选择最优划分
    attribute, gain = '', 0
    for a in attribute_set:
        tmp = Gain(dataset, a)
        if gain < tmp:
            attribute, gain = a, tmp
    node = Node(attribute)
    for v in attr_value_map[attribute]:
        subset = dataset[dataset[attribute] == v]
        if len(subset) == 0:
            most = Counter(dataset[label_tag]).most_common()[0][0]
            node.children[v] = Node(most)
        else:
            attribute_set.remove(attribute)
            node.children[v] = TreeGenerate(subset, attribute_set, attr_value_map)
            attribute_set.add(attribute)
    return node


if __name__ == '__main__':
    dataset = pd.DataFrame({
        "色泽": ["青绿", "乌黑", "乌黑", "青绿", "浅白", "青绿", "乌黑", "乌黑", "乌黑", "青绿", "浅白", "浅白", "青绿", "浅白", "乌黑", "浅白", "青绿"],
        "根蒂": ["蜷缩", "蜷缩", "蜷缩", "蜷缩", "蜷缩", "稍蜷", "稍蜷", "稍蜷", "稍蜷", "硬挺", "硬挺", "蜷缩", "稍蜷", "稍蜷", "稍蜷", "蜷缩", "蜷缩"],
        "敲声": ["浊响", "沉闷", "浊响", "沉闷", "浊响", "浊响", "浊响", "浊响", "沉闷", "清脆", "清脆", "浊响", "浊响", "沉闷", "浊响", "浊响", "沉闷"],
        "纹理": ["清晰", "清晰", "清晰", "清晰", "清晰", "清晰", "稍糊", "清晰", "稍糊", "清晰", "模糊", "模糊", "稍糊", "稍糊", "清晰", "模糊", "稍糊"],
        "脐部": ["凹陷", "凹陷", "凹陷", "凹陷", "凹陷", "稍凹", "稍凹", "稍凹", "稍凹", "平坦", "平坦", "平坦", "凹陷", "凹陷", "稍凹", "平坦", "稍凹"],
        "触感": ["硬滑", "硬滑", "硬滑", "硬滑", "硬滑", "软粘", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "硬滑"],
        "好瓜": ["是"] * 8 + ["否"] * 9
    })

    # 为了防止后面出现v不够的情况，比如在后面学习的时候数据集中没有"浅白"，导致少了一个分支
    attr_value_map = {k: np.unique(dataset[k]) for k in dataset.columns[:-1]}

    node = TreeGenerate(dataset, set(dataset.columns[:-1]), attr_value_map)
    print(node)
