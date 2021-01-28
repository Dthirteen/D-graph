import torch
import torch.nn as nn
from torch.nn import Softmax
from cc_attention import CrissCrossAttention
import math
#from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
# from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
# BatchNorm2d = SyncBN#functools.partial(InPlaceABNSync, activation='identity')
#from cc_attention import Self_Attn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv
from mmcv.cnn import constant_init, kaiming_init

import torch.nn as nn
from torch.nn import Softmax
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        #self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        #self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        #self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        #proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        #out = torch.bmm(proj_value,attention.permute(0,2,1) )
        #out = out.view(m_batchsize,C,width,height)
 
        #out = self.gamma*out + x
        #return out,attention
        return attention


class SSCDNonLModel(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(SSCDNonLModel, self).__init__()
        #self.num_Node=num_Node
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()


        self.b2=nn.BatchNorm2d(self.bands+chanel)
        #self.nlcon1=NonLocalBlock(300, 300, True)
        #self.gcn1=GCNtrans(300,1000)
        #self.bcat=nn.BatchNorm2d(300+300)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)
        self.nlcon2=CrissCrossAttention(CCChannel)
        self.nlcon3=CrissCrossAttention(CCChannel)
        self.bcat=nn.BatchNorm2d(CCChannel+CCChannel)
        self.con4=nn.Conv2d(CCChannel+CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        #self.b6=nn.BatchNorm2d(300+300)
        self.con6=nn.Conv2d(chanel+CCChannel, num_classes+1, 1, padding=0, bias=True)



    def forward(self, x):
        n = x.size(0)
        H=x.size(2)
        W=x.size(3)

        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)


        xx=self.b4(out2)
        nl2=self.nlcon2(xx)
        nl2=self.nlcon2(nl2)
        nl3=self.nlcon3(xx)
        nl3=self.nlcon3(nl3)
        nl2=(nl2+nl3)*0.7+xx
        #print(nl2.shape)
        out4=torch.cat((xx, nl2),1)
        #print(out4.shape)
        out4=self.bcat(out4)
        #print(out4.shape)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)



        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)


        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6

class SSCDNonLModel_ablation(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(SSCDNonLModel_ablation, self).__init__()
        #self.num_Node=num_Node
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()


        self.b2=nn.BatchNorm2d(self.bands+chanel)
        #self.nlcon1=NonLocalBlock(300, 300, True)
        #self.gcn1=GCNtrans(300,1000)
        #self.bcat=nn.BatchNorm2d(300+300)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)

        self.bcat=nn.BatchNorm2d(CCChannel+CCChannel)
        self.con4=nn.Conv2d(CCChannel+CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        #self.b6=nn.BatchNorm2d(300+300)
        self.con6=nn.Conv2d(chanel+CCChannel, num_classes+1, 1, padding=0, bias=True)



    def forward(self, x):
        n = x.size(0)
        H=x.size(2)
        W=x.size(3)

        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)


        xx=self.b4(out2)

        #print(nl2.shape)
        out4=torch.cat((xx, xx),1)
        #print(out4.shape)
        out4=self.bcat(out4)
        #print(out4.shape)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)



        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)


        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6

class SSCDNonLModel_gcn3(nn.Module):
    def __init__(self, num_classes, n_bands, chanel,threshold):
        super(SSCDNonLModel_gcn3, self).__init__()
        #self.num_Node=num_Node
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25
        self.threshold=threshold

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()


        self.b2=nn.BatchNorm2d(self.bands+chanel)
        #self.nlcon1=NonLocalBlock(300, 300, True)
        #self.gcn1=GCNtrans(300,1000)
        #self.bcat=nn.BatchNorm2d(300+300)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)

        # GCN
        self.trans=nn.Conv2d(CCChannel,1,1,padding=0,bias=True)
        #self.conv1 = GCNConv(dataset.num_features, 16, cached=True,normalize=not args.use_gdc)
        self.att_map=Self_Attn(CCChannel)
        self.conv1=GCNConv(1,int(CCChannel/2),cached=True)
        self.conv2=GCNConv(int(CCChannel/2),CCChannel,cached=True)

        # end
        #self.b4=nn.BatchNorm2d(CCChannel)
        #self.nlcon2=CrissCrossAttention(CCChannel)
        #self.nlcon3=CrissCrossAttention(CCChannel)
        self.bcat=nn.BatchNorm2d(CCChannel+CCChannel)
        self.con4=nn.Conv2d(CCChannel+CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        #self.b6=nn.BatchNorm2d(300+300)
        self.con6=nn.Conv2d(chanel+CCChannel, num_classes+1, 1, padding=0, bias=True)



    def forward(self, x):
        n = x.size(0)
        H=x.size(2)
        W=x.size(3)

        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)


        xx=self.b4(out2)
        
        #print("XX:",xx.shape)
        #xxx=xx.cpu()
        att_map=self.att_map(xx)
        x=self.trans(xx)
        #with torch.no_grad():

        adj=att_map.cpu()
        adj=adj.view(H*W,H*W)
        
        adj[adj<self.threshold]=0
        #adj[adj>=0.5]=1
        #print(adj)
        edge_index=adj.nonzero().t().contiguous()
        #edge_weight = adj[edge_index[0], edge_index[1]]
        
        #edge_weight.requires_grad_ = True
        #edge_weight =edge_weight.cuda()

        #print('edge_index:',edge_index.shape)
        edge_index=edge_index.cuda()
        out_gcn1=self.conv1(x.view(-1,1),edge_index)
        #out_gcn1=self.conv1(out_finalconv.view(-1, 1),edge_index)
        out_gcn1 = F.relu(out_gcn1)
        out_gcn1 = F.dropout(out_gcn1, training=self.training)
        out_gcn2=self.conv2(out_gcn1,edge_index)
        #x=F.log_softmax(out_gcn2, dim=1)
        #print(out_gcn2.shape)
        #print('out_gcn1:',out_gcn1.shape)
        #print('out_gcn2:',out_gcn2.shape)
        out_gcn2=out_gcn2.view(1,-1,H,W)  


        #nl2=(nl2+nl3)*0.7+xx
        out4=torch.cat((xx, out_gcn2),1)
        #print(out4.shape)
        #print(out_gcn2.shape)
        out4=self.bcat(out4)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)



        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)


        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6

class SSCDNonLModel_ngcn_1(nn.Module):
    def __init__(self, num_classes, n_bands, chanel,threshold):
        super(SSCDNonLModel_ngcn_1, self).__init__()
        #self.num_Node=num_Node
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25
        self.threshold=threshold

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()


        self.b2=nn.BatchNorm2d(self.bands+chanel)
        #self.nlcon1=NonLocalBlock(300, 300, True)
        #self.gcn1=GCNtrans(300,1000)
        #self.bcat=nn.BatchNorm2d(300+300)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)

        # GCN
        self.trans=nn.Conv2d(CCChannel,1,1,padding=0,bias=True)
        #self.conv1 = GCNConv(dataset.num_features, 16, cached=True,normalize=not args.use_gdc)
        self.att_map=Self_Attn(CCChannel)
        self.conv1=GCNConv(1,CCChannel,cached=True)
        

        # end
        #self.b4=nn.BatchNorm2d(CCChannel)
        #self.nlcon2=CrissCrossAttention(CCChannel)
        #self.nlcon3=CrissCrossAttention(CCChannel)
        self.bcat=nn.BatchNorm2d(CCChannel+CCChannel)
        self.con4=nn.Conv2d(CCChannel+CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        #self.b6=nn.BatchNorm2d(300+300)
        self.con6=nn.Conv2d(chanel+CCChannel, num_classes+1, 1, padding=0, bias=True)



    def forward(self, x):
        n = x.size(0)
        H=x.size(2)
        W=x.size(3)

        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)


        xx=self.b4(out2)
        
        #print("XX:",xx.shape)
        #xxx=xx.cpu()
        att_map=self.att_map(xx)
        x=self.trans(xx)
        #with torch.no_grad():

        adj=att_map.cpu()
        adj=adj.view(H*W,H*W)
        
        adj[adj<self.threshold]=0
        #adj[adj>=0.5]=1
        #print(adj)
        edge_index=adj.nonzero().t().contiguous()
        #edge_weight = adj[edge_index[0], edge_index[1]]
        
        #edge_weight.requires_grad_ = True
        #edge_weight =edge_weight.cuda()

        #print('edge_index:',edge_index.shape)
        edge_index=edge_index.cuda()
        out_gcn1=self.conv1(x.view(-1,1),edge_index)
        #out_gcn1=self.conv1(out_finalconv.view(-1, 1),edge_index)

        #print(out_gcn2.shape)
        #print('out_gcn1:',out_gcn1.shape)
        #print('out_gcn2:',out_gcn2.shape)
        out_gcn2=out_gcn1.view(1,-1,H,W)  


        #nl2=(nl2+nl3)*0.7+xx
        out4=torch.cat((xx, out_gcn2),1)
        #print(out4.shape)
        #print(out_gcn2.shape)
        out4=self.bcat(out4)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)



        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)


        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6


class SSCDNonLModel_ngcn_3(nn.Module):
    def __init__(self, num_classes, n_bands, chanel,threshold):
        super(SSCDNonLModel_ngcn_3, self).__init__()
        #self.num_Node=num_Node
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25
        self.threshold=threshold

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()


        self.b2=nn.BatchNorm2d(self.bands+chanel)
        #self.nlcon1=NonLocalBlock(300, 300, True)
        #self.gcn1=GCNtrans(300,1000)
        #self.bcat=nn.BatchNorm2d(300+300)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)

        # GCN
        self.trans=nn.Conv2d(CCChannel,1,1,padding=0,bias=True)
        #self.conv1 = GCNConv(dataset.num_features, 16, cached=True,normalize=not args.use_gdc)
        self.att_map=Self_Attn(CCChannel)
        self.conv1=GCNConv(1,int(CCChannel/4),cached=True)
        self.conv2=GCNConv(int(CCChannel/4),int(CCChannel/2),cached=True)
        self.conv3=GCNConv(int(CCChannel/2),CCChannel,cached=True)
        

        # end
        #self.b4=nn.BatchNorm2d(CCChannel)
        #self.nlcon2=CrissCrossAttention(CCChannel)
        #self.nlcon3=CrissCrossAttention(CCChannel)
        self.bcat=nn.BatchNorm2d(CCChannel+CCChannel)
        self.con4=nn.Conv2d(CCChannel+CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        #self.b6=nn.BatchNorm2d(300+300)
        self.con6=nn.Conv2d(chanel+CCChannel, num_classes+1, 1, padding=0, bias=True)



    def forward(self, x):
        n = x.size(0)
        H=x.size(2)
        W=x.size(3)

        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)


        xx=self.b4(out2)
        
        #print("XX:",xx.shape)
        #xxx=xx.cpu()
        att_map=self.att_map(xx)
        x=self.trans(xx)
        #with torch.no_grad():

        adj=att_map.cpu()
        adj=adj.view(H*W,H*W)
        
        adj[adj<self.threshold]=0
        #adj[adj>=0.5]=1
        #print(adj)
        edge_index=adj.nonzero().t().contiguous()
        #edge_weight = adj[edge_index[0], edge_index[1]]
        
        #edge_weight.requires_grad_ = True
        #edge_weight =edge_weight.cuda()

        #print('edge_index:',edge_index.shape)
        edge_index=edge_index.cuda()
        out_gcn1=self.conv1(x.view(-1,1),edge_index)
        #out_gcn1=self.conv1(out_finalconv.view(-1, 1),edge_index)
        out_gcn1 = F.relu(out_gcn1)
        out_gcn1 = F.dropout(out_gcn1, training=self.training)
        out_gcn2=self.conv2(out_gcn1,edge_index)
        out_gcn2 = F.relu(out_gcn2)
        out_gcn2 = F.dropout(out_gcn2, training=self.training)
        out_gcn3=self.conv3(out_gcn2,edge_index)
        #x=F.log_softmax(out_gcn2, dim=1)
        #print(out_gcn2.shape)
        #print('out_gcn1:',out_gcn1.shape)
        #print('out_gcn2:',out_gcn2.shape)
        out_gcn3=out_gcn3.view(1,-1,H,W) 


        #nl2=(nl2+nl3)*0.7+xx
        out4=torch.cat((xx, out_gcn3),1)
        #print(out4.shape)
        #print(out_gcn2.shape)
        out4=self.bcat(out4)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)



        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)


        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6
        ## gcnet 版本

import networkx as nx
import matplotlib.pyplot as plt

def draw(edge_index, name=None):
    G = nx.Graph(node_size=15, font_size=8)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        G.add_edge(i, j)
    plt.figure(figsize=(200, 140)) # 设置画布的大小
    nx.draw_networkx(G)
    plt.savefig('{}.png'.format(name if name else 'path'))



class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att'):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
       
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool

        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True



    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context = self.softmax(context_mask)#softmax操作

        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)



        return context