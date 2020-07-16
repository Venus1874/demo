#搭建卷积神经网络手写体
import torch 
import torchvision#包含数据库，包括图片数据库
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

#Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False #没有下载好数据就设为true,已将下载好就设为Flase

train_data = torchvision.datasets.MNIST(
    root = './mnist',
    train= True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
'''print(train_data.train_data.size())   #(60000, 28,28)
print(train_data.train_labels.size())  #(60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i'%train_data.train_labels[0])
plt.show()'''

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]#取前两千张图

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(                #图片的维度和宽高(1,28,28)
                in_channels=1,
                out_channels=16,      #提取了16个特征
                kernel_size=5,
                stride=1,             #步长
                padding=2             #边界补零if stride = 1,padding = (kernel_size-1)/2=(5-1)/2可以使得输入与输出长宽一样
            ),                        #->(16,28,28)
            nn.ReLU(),                #->(16,28,28)
            nn.MaxPool2d(kernel_size=2),#->(16,14,14)
        )
        self.conv2 = nn.Sequential(    #->(16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),#接受进来16层，再加工成32层，卷积核5，stride,padding  #->(32,14,14)
            nn.ReLU(),                 #->(32,14,14)
            nn.MaxPool2d(2)            #->(32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)               #(batch, 32, 7, 7)
        x = x.view(x.size(0),-1)        #(batch, 32 * 7 * 7) -1意思就是把32, 7, 7->32 * 7 * 7
        output = self.out(x)
        return output

cnn = CNN()
#print(cnn) #net architerture
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR) #optimze all cnn parameters
loss_func = nn.CrossEntropyLoss()                     #the target label is not one-hotted

#training and testing
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        output = cnn(b_x)             # cnn output 
        loss = loss_func(output, b_y) #cross entropy loss 
        optimizer.zero_grad()         #clear gradients for this training step
        loss.backward()               #backpropagation,compute gradients
        optimizer.step()              #apply gradients

        if step % 100 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = torch.sum(pred_y == test_y).data.float() / test_y.size(0)   
            print('Epoch: ', epoch, '| train loss:%.4f '%loss.item(),'| test accuracy: %.2f'%accuracy)

#print 10 predictions from test data 
torch.save(cnn,'cnn.pkl')
cnn2=torch.load('cnn.pkl')
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')