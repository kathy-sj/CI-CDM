
import torch
import numpy as np


class CICDM(torch.nn.Module):#构建模型

    def __init__(self,s_n, k_n, N_out):
        super(CICDM, self).__init__()
        self.k_n = k_n
        self.k_n_select = 3
        self.N_out = N_out
        self.nVars = 2 ** self.k_n_select - 2 #需学习的模糊测度个数
        self.emb_num = s_n  # 学生数
        self.stu_dim =30

        self.student_emb = torch.nn.Embedding(self.emb_num, self.stu_dim)
         # alpha——MLP网络
        self.prednet_input_len1, self.prednet_input_len2 = 128,256
        self.input_layer1 = torch.nn.Sequential(
            torch.nn.Linear( self.stu_dim,self.prednet_input_len1),
            torch.nn.ReLU(True) )        #self.prednet_len1

        self.input_layer3 = torch.nn.Sequential(torch.nn.Linear(self.prednet_input_len1, self.k_n))


        #CHI输出网络
        # #网络节点数 changeable
        self.prednet_len = N_out
        self.prednet_len1, self.prednet_len2 = 256,128
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(self.prednet_len,self.prednet_len1), torch.nn.ReLU(True))#self.prednet_len1
        self.drop_1 = torch.nn.Dropout(p=0.5)
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(self.prednet_len1,self.prednet_len2), torch.nn.ReLU(True)) #self.prednet_len1
        self.drop_2 = torch.nn.Dropout(p=0.5)
        self.layer3 = torch.nn.Sequential(torch.nn.Linear(self.prednet_len2,self.N_out ))
        self.device = torch.device(('cuda:1') if torch.cuda.is_available() else 'cpu')
        # torch.nn.init.xavier_normal_(self.student_emb.weight)
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)

        dummy = (1. /self.k_n_select) * torch.ones((self.nVars, self.N_out), requires_grad=True)#初始化模糊测度值
        self.vars = torch.nn.Parameter(dummy)#参数  等同于exer_embing

        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.k_n_select)

        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]  # 一个list,其中的元素均为tensor
        self.subset = [torch.tensor(x) for x in self.subset]
        self.q = torch.from_numpy(np.loadtxt("./data/q_m_select.csv", delimiter=","))


    def forward(self, stu_id, exer_id, kn_emb):  # choquet 积分学习

        stu_emb = self.student_emb(stu_id)#知识掌握水平嵌入向量
        self.inputs_select = torch.sigmoid (stu_emb)

        inputs_select =  self.inputs_select.unsqueeze(1)#扩展维度

        inputs1= inputs_select.expand(stu_emb.shape[0],self.N_out,self.k_n)#知识点的掌握水平扩展到所有试题  #64,13
       # print(np.shape(inputs1))

        #根据Q矩阵筛选出每道题对应的知识点
        q=self.q.unsqueeze(0)
        q=q.expand(inputs1.shape[0],self.q.shape[0],self.q.shape[1])
        index=q.to(torch.uint8).bool()
        inputs = inputs1[index].reshape(inputs1.shape[0],self.N_out,-1 )#剔除每道题不涉及的知识点
        inputs =inputs.to (torch.float32)#inputs是每道题对应知识的知识掌握程度

        sortInputs, sortInd = torch.sort(inputs, 2, True)  # 对输入的值（知识掌握程度）降序排序获得sortInputs对应值, sortInd对应序号
        

        M, N = inputs.shape[0],inputs.shape[2]  # 每一bath的数量和对应的知识点数
        zz = torch.zeros(M, inputs.shape[1], 1).to(self.device)#生成0向量0
        sortInputs = torch.cat((sortInputs, zz), 2)#排序后的知识掌握程度最后一列增加0元素
        # sortInputs.is_cuda
        sortInputs = sortInputs[:, :,:-1] - sortInputs[:,:,1:]#对知识掌握程度变换
        out = torch.cumsum(torch.pow(2, sortInd), 2) - torch.ones(1, dtype=torch.int).to(self.device)  # cumsum返回维度dim中输入元素的累计和。

        data = torch.zeros((M, inputs.shape[1],self.nVars + 1)).to(torch.float32).to(self.device)


        alpha_input = data.scatter(2,out,sortInputs)#知识掌握程度向量放入对应的位置，与FM计算的位置
        

        self.FM1 = self.chi_nn_vars(self.vars)#FM学习

        self.FM = self.FM1.expand (inputs.shape[0],self.N_out,self.nVars+1)#扩展维度
        C =torch.sum (torch.mul(alpha_input, self.FM),axis=2).to(torch.float32)#CHI计算
        #C = torch.sum (inputs,axis=2).to(torch.float32)


        output_c = self.drop_1(self.layer1(C))
        output_c = self.drop_2(self.layer2(output_c))
        output = torch.sigmoid(self.layer3(output_c))

        return output




    def chi_nn_vars(self, chi_vars):#模糊测度的学习
        chi_vars = torch.abs(chi_vars)

        FM = chi_vars[None, 0, :]
        # print(FM)
        for i in range(1, self.nVars):
            indices = subset_to_indices(self.subset[i])#
            if (len(indices) == 1):#单个知识点集合
                FM = torch.cat((FM, chi_vars[None, i, :]), 0)  # 按维度0把两个张量拼接
            else:#多个知识点集合
                #         ss=tf.gather_nd(variables, [[1],[2]])
                maxVal, _ = torch.max(FM[indices, :], 0)#取子集中的最大值，保证弱单调性
                temp = torch.add(maxVal, chi_vars[i, :])  # chi_vars[i,:]的值添加进maxval一维张量
                FM = torch.cat((FM, temp[None, :]), 0)  # temp[None,:]二维张量

        FM = torch.cat([FM, torch.ones((1, self.N_out)).to(self.device)], 0)
        FM = torch.min(FM, torch.ones(1).to(self.device))
        FM = FM.t()

        return FM

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.layer1.apply(clipper)
        self.layer2.apply(clipper)
        self.layer3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        # alpha=self.inputs_select.detach().numpy()
        return stat_emb.data

    def get_exer_params(self):
        FM_learned = (self.chi_nn_vars(self.vars).cpu()).detach().numpy()
        # k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        # e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # return k_difficulty.data, e_discrimination.data
        return FM_learned.data



# Convert decimal to binary string
def sources_and_subsets_nodes(N):#模糊测度二进制表示
    str1 = "{0:{fill}" + str(N) + "b}"
    a = []
    for i in range(1, 2 ** N):
        # print(str1.format(i, fill='0'))  # 1-001，2-010.3-011
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))


    def node_subset(node, sourcesInNodes):
        return [node - 2 ** (i) for i in sourcesInNodes]


    def string_to_integer_array(s, ch):
        N = len(s)
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]  # i为索引，ltr为对应的取值



    for j in range(len(a)):
        # index from right to left
        idxLR = string_to_integer_array(a[j], '1')
        # print(idxLR)
        sourcesInNode.append(idxLR)
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))  # 变为集合set对象
        subset.append(node_subset(j, idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
