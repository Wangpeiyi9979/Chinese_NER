#encoding:utf-8
"""
@Time: 2020/11/13 15:28
@Author: Wang Peiyi
@Site : 
@File : create_tool_data.py
"""

# 创建label.txt
import json
json_datas = []
all_labels = set()
origin_datas = open('./dataset/train.json')
for data in origin_datas:
    data = json.loads(data)
    labels = data['label'].keys()
    for label in labels:
        all_labels.add('B-{}'.format(label))
        all_labels.add('I-{}'.format(label))
with open('dataset/tool_data/label.txt', 'w') as f:
    f.write('O\n')
    for label in all_labels:
        f.write('{}\n'.format(label))
