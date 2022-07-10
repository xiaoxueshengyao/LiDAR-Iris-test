import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import math

from functools import reduce
 
def str2int(s):
    def fn(x,y):
        return x*10+y
    def char2num(s):
        return {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]
    return reduce(fn,map(char2num,s))

sys.path.append("..")

seq = "08"
N_seq = 4071
N = 0


#read kitti pose
traj = np.loadtxt(seq +".txt")

# ./demo [lidar iris result]
dist0 = np.loadtxt("./test_res"+seq+".txt")
print("dist0: ")
print(dist0[1320])
gt = {}
gt_mat = [ [0] * N_seq for i in range(N_seq)]
gt_mat = np.array(gt_mat)

for line in open("./gt"+seq+".txt", "r"):
    if line.strip():
        sl = line.split()
        if len(sl) >= 2:            
            for i in range(1,len(sl)-1):
                if str2int(sl[0]) - str2int(sl[i]) > 300:
                    gt[sl[0]] = 1
                    # print(sl[0],sl[i])
                    gt_mat[str2int(sl[0])-1][str2int(sl[i])-1] = 1
                    if i<=1:
                        N = N+1                  
                else:
                    gt[sl[0]] = 0

print(N)
print(gt_mat)
# np.savetxt("gt_mat05.txt",gt_mat,fmt='%d')

x_cord = traj[:,3]
z_cord = traj[:,11]

#dist 0
th0 = []
pre0 = []
rec0 = []
dist0_min = dist0[:, 2].min()
dist0_max = dist0[:, 2].max()

print(dist0_min, dist0_max)


# p = 0
# tp = 0
# test = 0
# print("dist0 shape0: ",dist0.shape[0])
# for i in range(0,dist0.shape[0]):
#     if dist0[i][2] <= 0.13 and dist0[i][1] != 0:
#         # if dist0[i][3] == 0:
#         #     test = test+1
#         #     print(dist0[i])
#         p=p+1
#         # print(dist0[i][0],dist0[i][1])
#         # print(gt_mat[dist0[i][0].astype(np.int)-1][dist0[i][1].astype(np.int)]-1)
#         if gt_mat[dist0[i][0].astype(np.int)-1][dist0[i][1].astype(np.int)-1] == 1:
#             tp = tp+1

# pre = tp*1.0/p
# rec = tp*1.0/N
# print("test: ",test)
# print("precision :",pre)
# print("recall is :",rec)

for i in np.arange(dist0_min+0.05, dist0_max + (dist0_max-dist0_min) * 1.0 /100, (dist0_max-dist0_min) * 1.0 /100):
    print(i)
    tp = 0
    p = 0
    for j in range(0, dist0.shape[0]):
        if dist0[j][2] <= i and dist0[j][1] > 0:
            # print("dist0 dis: ",dist0[j][2])
            p = p+1
            m = dist0[j][0].astype(int)
            n = dist0[j][1].astype(int)
            # print("m,n",m,n)  
            if gt_mat[m-1][n-1] == 1:
                tp = tp+1
    
    re = tp * 1.0 / N
    pr = tp * 1.0 / p
    print("recall",re)
    print("precision", pr)
    th0.append(i)
    rec0.append(re)
    pre0.append(pr)

thres = 0
for i in range(len(th0)-1):
    print([th0[i],pre0[i],rec0[i]])
    if pre0[i]==1.0 and pre0[i+1]!=1.0:
        thres = th0[i]
        break 


#draw p-r curve
#coding:utf-8
fig1 = plt.figure(1) # create figure 1
plt.title('Precision/Recall Curve',fontsize=20)# give plot a title
plt.xlabel('Recall', fontsize=20)# make axis labels
plt.ylabel('Precision',fontsize=20)
plt.tick_params(labelsize=18)
plt.plot(rec0, pre0,  "r", label = "ScanContext", linewidth=3.0)
plt.legend(loc="lower left", fontsize=20)


fig2 = plt.figure(2)
plt.title('trajectory',fontsize=20)# give plot a title
plt.xlabel('x', fontsize=20)# make axis labels
plt.ylabel('z',fontsize=20)
plt.tick_params(labelsize=18)
plt.plot(x_cord, z_cord,  "k", linewidth=1.0)

for i in range(len(dist0[:,0])):
    if gt[str(int(dist0[i][0]))]:
        index = int(int(dist0[i][0])-1)
        plt.scatter(x_cord[index], z_cord[index], c="g",alpha=0.2)
    if dist0[i][2] <= thres and dist0[i][3] == 1:
        index = int(dist0[i][0]-1)
        plt.scatter(x_cord[index], z_cord[index], c="r")
    if dist0[i][2] <= thres and dist0[i][1] >0 and dist0[i][3] == 0:
        index = int(dist0[i][0]-1)
        plt.scatter(x_cord[index], z_cord[index], c="b")

plt.show()
