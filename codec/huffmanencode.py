import torch
import os
import copy
import numpy as np
from bitstring import BitArray, BitStream, ConstBitStream


#open('video.m2v', 'wb').write(s.bytes)
#s = ConstBitStream(filename='test/test.m1v')

max_num_table={
    "uint8": 256,
    "uint12": 4096
}

offset_table={
    "uint8": 127,
    "uint12": 2048
}
class Node:
    def __init__(self,c):
        self.count=c
        self.left=-1
        self.right=-1
        self.removed=False

def findMinNode(nodes, length):
    index=-1
    for i in range(length):
        if (index==-1 or nodes[i].count<nodes[index].count) and (not nodes[i].removed) and nodes[i].count>0:
            index=i

    if index !=-1:
        nodes[index].removed =1

    return index

def build_tree(nodes, counts,num_type='uint12'):
    max_num = max_num_table[num_type]

    for i in range(max_num):
        nodes.append(Node(counts[i]))

    length=max_num

    while True:
        l= findMinNode(nodes,length)
        if l==-1:
            break

        r = findMinNode(nodes, length)
        if r == -1:
            break

        nodes.append(Node(0))
        nodes[length].left = l
        nodes[length].right = r
        nodes[length].count = nodes[l].count + nodes[r].count
        nodes[length].removed = False
        length +=1

    return length

def build_table(nodes,pos,bits,table):
    l = nodes[pos].left
    r = nodes[pos].right
    if (nodes[pos].left == -1 and nodes[pos].right == -1):
        table[pos] = bits
        return

    build_table(nodes, r, bits + "1", table)
    build_table(nodes, l, bits + "0", table)

def huffman_encode( RES, save_path,num_type='uint12' ):
    # default  int12
    print(" start huffman encode using type", num_type,'at',save_path)
    max_num=max_num_table[num_type]
    offset=offset_table[num_type]

    data=RES+offset
    # 感觉好像没有必要分channel分别encode ？？？到时候试一下那种方式的编码好？
    #density 和 color 应该分开
    data_shape=data.size()
    data=data.reshape(-1)

    data=data.cpu()
    counts = torch.zeros(max_num, device=data.device)

    for i in range(data.size(0)):
        counts[int(data[i])]+=1

    print("counts",torch.max(counts,0))
    print("counts percent", torch.max(counts)/torch.sum(counts))
    nodes=[]
    length=build_tree(nodes, counts,num_type)

    table=["" for _ in range(max_num)]
    build_table(nodes, length - 1, "", table)

    table_path = save_path+"_table"
    #这个table感觉存的不是很好，是因为可以假设一开始就传吗？
    with open(table_path, 'w') as f:
        for i in range(max_num):
            if len(table[i])==0:
                f.write("2\n")
            else:
                f.write(f"{table[i]}\n")

    total_bit_length = 0
    for  i in range(max_num):
        total_bit_length += counts[i] * len(table[i])

    total=""
    for i in range(data.size(0)):
        total+=table[int(data[i])]

    if total_bit_length %8 !=0:
        times = total_bit_length // 8 +1

        for i in range(0,int(8 * times - total_bit_length)):
            total+='0'

        total_bit_length=8*times

    print(total_bit_length)

    total=BitArray('0b'+total)
    open(save_path+"_data.ncrf", 'wb').write(total.bytes)

    return total_bit_length







