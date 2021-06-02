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


def TreeGenerate(dataset, attribute_set, label_tag="好瓜"):
    labels = np.unique(dataset[label_tag])
    if len(labels) == 1:
        # 样本全部属于同一类别
        return Node(labels[0])
    if len(attribute_set) == 0:
        # 当前属性集为空
        most = Counter(dataset[label_tag]).most_common()[0][0]
        return Node(most)
    # 选择最优划分
    tag, gain = '', 0
    for attribute in attribute_set:
        tmp = Gain(dataset, attribute)
        if gain < tmp:
            tag, gain = attribute, tmp
    node = Node(tag)
    for v in np.unique(dataset[tag]):
        subset = dataset[dataset[tag] == v]
        if len(subset) == 0:
            most = Counter(dataset[label_tag]).most_common()[0][0]
            node.children[v] = Node(most)
        else:
            attribute_set.remove(tag)
            node.children[v] = TreeGenerate(subset, attribute_set)
            attribute_set.add(tag)
    return node


if __name__ == '__main__':
    dataset = pd.DataFrame({
        "色泽": ["青绿", "乌黑", "乌黑", "青绿", "浅白", "青绿", "乌黑", "乌黑", "乌黑", "青绿", "浅白", "浅白", "青绿", "浅白", "乌黑", "浅白", "青绿"],
        "根蒂": ["蜷缩", "蜷缩", "蜷缩", "蜷缩", "蜷缩", "稍蜷", "稍蜷", "稍蜷", "稍蜷", "硬挺", "硬挺", "蜷缩", "稍蜷", "稍蜷", "稍蜷", "蜷缩", "蜷缩"],
        "敲声": ["浊响", "沉闷", "浊响", "沉闷", "浊响", "浊响", "浊响", "浊响", "沉闷", "清脆", "清脆", "浊响", "浊响", "沉闷", "浊响", "浊响", "沉闷"],
        "纹理": ["清晰", "清晰", "清晰", "清晰", "清晰", "清晰", "稍糊", "清晰", "稍糊", "清晰", "模糊", "模糊", "稍糊", "稍糊", "清晰", "模糊", "稍糊"],
        "脐部": ["凹陷", "凹陷", "凹陷", "凹陷", "凹陷", "稍凹", "稍凹", "稍凹", "稍凹", "平坦", "平坦", "平坦", "凹陷", "凹陷", "稍凹", "平坦", "稍凹"],
        "触感": ["硬滑", "硬滑", "硬滑", "硬滑", "硬滑", "软粘", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "硬滑"],
        "好瓜": ["是" for _ in range(8)] + ["否" for _ in range(9)]
    })

    node = TreeGenerate(dataset, {"色泽", "根蒂", "敲声", "纹理", "脐部", "触感"})
    print(node)
