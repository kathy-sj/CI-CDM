                                                                                                                                                                               # -*- coding: utf-8 -*-
"""
Created on 2022.01.07
@author: zsj
"""
import torch
import numpy as np
from data_loader import TrainDataLoader, ValTestDataLoader
from CICDM_model import CICDM

from sklearn.metrics import roc_auc_score,  mean_squared_error, mean_absolute_error, f1_score
# Convert decimal to binary string


def train(num_epochs):
    loss_list = []
    epochs= num_epochs
    # print(s_n, k_n, N_out)
    net = CICDM(s_n, k_n, N_out)
    net = net.to(device)


    for t in range(0,epochs):
        print('epoch:', t)
        # Forward pass: Compute predicted y by passing x to the model
        data_loader = TrainDataLoader()
        data_loader.reset()
        batch_count =0
        learning_rate =1e-3
        pred_all, label_all = [], []
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        correct_count, exer_count = 0, 0
        while not data_loader.is_end():
            batch_count += 1

            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
           # print(labels)
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
                device), input_knowledge_embs.to(device), labels.to(device)
            label_one_hot = torch.nn.functional.one_hot(input_exer_ids, num_classes=N_out).to(torch.double)

            # labels1= torch.mul(label_one_hot,labels.view(-1,1))

            y_pred = net.forward(input_stu_ids,input_exer_ids ,input_knowledge_embs)
            
            y_pred1 = torch.mul(label_one_hot, y_pred)
            y_pred2 =torch.sum (y_pred1,axis=1).view(-1,1)    #单个值
            for i in range(len(labels)):
                if (labels[i] == 1 and y_pred2[i] > 0.5) or (labels[i] == 0 and y_pred2[i] < 0.5):
                    correct_count += 1
            exer_count += len(labels)
            

            pred_all += y_pred2.to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()

            #nll=loss 此时label需要是整数作为标签
            y_pred_0 = torch.ones(y_pred2.size()).to(device) - y_pred2
            output = torch.cat((y_pred_0, y_pred2), 1)
          #  print(len(y_pred1))
            loss = criterion(output, labels.long())
            #loss = criterion(y_pred1, labels1)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.apply_clipper()

        pred_score = np.greater(pred_all,0.5)
        pred_score = pred_score.astype(int)
        train_f1_sum = f1_score(label_all, pred_score)
        train_mae_sum = mean_absolute_error(label_all, pred_all)

        print('epoch= %d,  f1= %f,训练集mae= %f' % (t , train_f1_sum ,train_mae_sum))
        with open('data/result/model_train.txt', 'a', encoding='utf8') as f:
            f.write('epoch= %d,  f1= %f,mae= %f\n' % (t , train_f1_sum ,train_mae_sum))



        loss_list.append(loss.tolist())
        print('损失：', loss.tolist())


        save_snapshot(net, 'data/model3/model_epoch' + str(t + 1))
        val_auc ,val_f1, val_mae = validate(net, t)
        print('epoch= %d,验证集auc= %f, 验证集f1= %f,验证集mae= %f' % (t, val_auc ,val_f1, val_mae))
        with open('data/result3/model_val.txt', 'a', encoding='utf8') as f:
            f.write('epoch= %d,验证集auc= %f,验证集f1= %f,验证集mae= %f\n' % (t, val_auc,val_f1, val_mae))

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

def validate(model, epoch):
    data_loader = ValTestDataLoader('test')
    net =CICDM(s_n, k_n, N_out)
    # print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()


    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    error_list_val=[]
    error_list1_val = []
    # a_k = np.loadtxt("./data/D6/alpha_5910.csv", dtype=float, delimiter=',')
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
             device), input_knowledge_embs.to(device), labels.to(device)
        label_one_hot = torch.nn.functional.one_hot(input_exer_ids, num_classes= N_out).to(torch.double)
        labels1 = torch.mul(label_one_hot, labels.view(-1, 1))

        # y_pred = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        y_pred = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        y_pred1 = torch.mul(label_one_hot, y_pred)
        y_pred2 = torch.sum(y_pred1, axis=1).view(-1, 1)  # 单个值
        for i in range(len(labels)):
            if (labels[i] == 1 and y_pred2[i] > 0.5) or (labels[i] == 0 and y_pred2[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)

        pred_all += y_pred2.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()


        
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    auc = roc_auc_score(label_all, pred_all)

    pred_score = np.greater(pred_all,0.5)
    pred_score = pred_score.astype(int)
    val_f1_sum = f1_score(label_all, pred_score)
    val_mae_sum = mean_absolute_error(label_all, pred_all)


    return auc,val_f1_sum, val_mae_sum


if __name__=="__main__":
    print(torch.cuda.is_available() ) # 是否有已经配置好可以使用的GPU (若True则有)

    print(torch.cuda.device_count()  )
    # 学生人数
    s_n = 5625
    # 知识点数
    k_n = 30
    # 题目数
    N_out = 22

    criterion = torch.nn.CrossEntropyLoss()

    num_epochs =200

    #定义运行环境
    device = torch.device(('cuda:1') if torch.cuda.is_available() else 'cpu')

    train(num_epochs)


        












        

