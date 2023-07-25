
'''

Resnet models

                          NOTE: only layers required are retained and fine-tuned.
                                separate classes for resnet with and without hook for GradCAM.

'''
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.features_conv = self.vgg.features[:36]

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7, end2end=True):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
       
        bs = x.size(0)
        f = x

        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        
        f = self.layer1(f)
        #print('layer1: ',f.size())
        f = self.layer2(f)
        #print('layer2: ',f.size())
        f = self.layer3(f)
        feature = f.view(bs, -1)
        #print('layer4: ',f.size())
        f = self.layer4(f)
        #print('layer4: ',f.size())

        #hook for gradcam
        #h = f.register_hook(self.activations_hook)

        #f = self.avgpool(f)
        
        #f = f.squeeze(3).squeeze(2)
        #return f
        return  F.normalize(f) #f

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
        
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

class ResNetCAM(nn.Module):

    def __init__(self, block, layers, num_classes=7, end2end=True):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNetCAM, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
       
        bs = x.size(0)
        f = x

        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        
        f = self.layer1(f)
        #print('layer1: ',f.size())
        f = self.layer2(f)
        #print('layer2: ',f.size())
        f = self.layer3(f)
        feature = f.view(bs, -1)
        #print('layer4: ',f.size())
        f = self.layer4(f)
        #print('layer4: ',f.size())

        #hook for gradcam
        #h = f.register_hook(self.activations_hook)

        #f = self.avgpool(f)
        
        #f = f.squeeze(3).squeeze(2)
        #return f
        return  F.normalize(f) #f

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
        
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
        #return self.net(x)

class RN18(ResNetCAM):
  def __init__(self, block, layers,  pre_net, pre_cls, device = 'cuda', end2end=True):
    
    super(RN18, self).__init__(block, layers, num_classes=7, end2end=True)
    self.net  = resnet18CAMNet()
    self.net = load_base_model(self.net, pre_net)
    
    #self.net = load_base_model(self.net, pre_trained)
    #self.net  = nn.DataParallel(self.net).to(device)
    self.classifier = Classifier(512, 7)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    self.classifier = load_class(self.classifier, pre_cls)
    
    #self.features_conv = list(self.net.modules())
    self.gradients = None

  def activations_hook(self, grad):
        self.gradients = grad
        
  def forward(self, x):
        x = self.net(x)
        #print(x.shape)
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avgpool(x)
        #x = x.view((1, -1))
        #print(x.shape)
        x = x.squeeze(3).squeeze(2)
        #x = np.squeeze(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        #f.normalize(f)
        #x = x.item()
        return x#F.normalize(x)

  def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
  def get_activations(self, x):
        #return self.features_conv(x)
        return self.net(x)
  



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18CAM(pre1, pre2):
    """Constructs a ResNet-18 model with hookd for GradCAM.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RN18(BasicBlock, [2, 2, 2, 2], pre1, pre2)
    
    return model

def resnet18CAMNet():
    """Constructs a ResNet-18 model with hookd for GradCAM.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetCAM(BasicBlock, [2, 2, 2, 2])
    
    return model

def load_base_model(model, file1 = '/content/drive/MyDrive/ijba_res18_naive.pth.tar'): #load pretrained MSCeleb-1M     
   net_dict = torch.load(file1)
   if file1 == '/content/drive/MyDrive/ijba_res18_naive.pth.tar':
      pretrained_state_dict = net_dict['state_dict']
   else:
      pretrained_state_dict = net_dict
   model_state_dict = model.state_dict()
   #print("Original weights!!!\n", model.conv1.weight)
   for key in pretrained_state_dict:
      if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ) :  
        pass    
      else: 
        if file1 == '/content/drive/MyDrive/ijba_res18_naive.pth.tar':
          ch = '.'
          k= key[key.index(ch)+1:]
        else:
          k = key    
                   
        model_state_dict[k] = pretrained_state_dict[key]
   
   model.load_state_dict(model_state_dict, strict = False)
   #print("NEW weights!!!\n", model.conv1.weight)
   return model

def load_class(classifier, file ):
   if file == '/content/drive/MyDrive/ijba_res18_naive.pth.tar':
      return classifier
   net_dict = torch.load(file)
   
   #print("Original weights!!!\n", classifier.fc.bias)
   if file == '/content/drive/MyDrive/ijba_res18_naive.pth.tar':
      pretrained_state_dict = net_dict['state_dict']
   else:
      pretrained_state_dict = net_dict
   model_state_dict = classifier.state_dict()
   prefix = 'classifier.'
   n_clip = len(prefix)
   adapted_dict = {k[n_clip:]: v for k, v in pretrained_state_dict.items()
                if k.startswith(prefix)}
   for key in pretrained_state_dict:
      if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='fc.weight') | (key=='fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ): 
        #print("True") 
        if file == '/content/drive/MyDrive/ijba_res18_naive.pth.tar':
          ch = '.'
          #print("HI")
          k= key[key.index(ch)+1:]
          #k = key
        else:
          #print("hi")
          ch = '.'
          #print("HI")
          #k= key[key.index(ch)+1:]
          k = key           
        model_state_dict[k] = pretrained_state_dict[key]

   classifier.load_state_dict(model_state_dict)
   #print("New weights!!!\n", classifier.fc.bias)
   return classifier

class Classifier(nn.Module):
  def __init__(self, input , out):
    super(Classifier, self).__init__()
    self.fc = nn.Linear(input, out)
  
  def forward(self, x):
    output = self.fc(x)
    probs = F.softmax(output)
    return probs
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" )
    model = torch.nn.DataParallel(resnet18(end2end= False,  pretrained= False)).to(device)
    print('net #',count_parameters(model))
    x = torch.rand(2,  3, 224,224).to(device)
    f = model(x)
    print(f.size())
if __name__=='__main__':
    main()
