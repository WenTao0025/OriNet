import torch
from torch import nn
from torchvision import models
class QueryEncoder(nn.Module):
    print("queryencoder")
    def __int__(self,**kwargs):
        super(QueryEncoder, self).__int__(**kwargs)
        self.resnet = models.resnet50(pretrained=True)
        fc_feature = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(fc_feature * 1),
            nn.Linear(fc_feature*1,self.dim)
        )
    def forward(self,input):
        embeddings = self.resnet(input)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

class RenderEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.dim = out_dim
        self.resnet = models.resnet50(weights=None)

        fc_feature = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(fc_feature * 1),
            nn.Linear(fc_feature * 1, self.dim)
        )
    def forward(self, input):
        embeddings = self.resnet(input)
        embeddings = torch.nn.functional.normalize(embeddings, p = 2, dim = 1)
        return embeddings

if __name__ == '__main__':
    pre = torch.load("./test/pretrained/resnet50-11ad3fa6.pth")
    r = RenderEncoder()
    print(r.resnet)
    # r.resnet.load_state_dict(pre)
    # print(r.children())
    # for name,parameters in r.named_parameters():
    #     # print("a")
    #     print(name)

