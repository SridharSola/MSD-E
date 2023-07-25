import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from kd_student_dataset import *
import resnet
import utils

#Creating parser to store arguments to pass to main 

parser = argparse.ArgumentParser(description='CKnowledge Distillation')

# Optimization options
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=96, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--workers', type=int, default=16,
                        help='num of workers to use')
parser.add_argument('--folds', default=10, type=int, metavar='N', help='cross validation folds')

parser.add_argument('--lamda', type=int, default=100)


# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

#Device options
parser.add_argument('--gpu', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#Method options
parser.add_argument('--train-iteration', type=int, default=800,
                        help='Number of iteration per epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=1000, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')


#Data
parser.add_argument('--classes', type=int, default=7)

parser.add_argument('--root', type=str, default='/content/drive/MyDrive/Student_Dataset_FER',
                        help="root path to train data directory")

parser.add_argument('--subroot', type=list, default=['/MSD-E', '/MSD-ME'])

parser.add_argument('--model-dir', default='/content/drive/MyDrive/Student_Dataset_FER/Checkpoint', type=str)

parser.add_argument('--image-list', default='/content/drive/MyDrive/Student_Dataset_FER/Paired_files.txt', type=str, help='')

parser.add_argument('--logfile', default='/content/drive/MyDrive/Student_Dataset_FER/StuLog.txt', type=str)

#args = parser.parse_args()
args = parser.parse_args(" ".split())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To use GPU


best_acc = 0
def main(args):
    global best_acc

    #Data
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    imagesize = args.imagesize
    train_transform = A.Compose([
                                      A.HorizontalFlip(), A.HueSaturationValue(), A.RandomContrast(), 
                                      A.ShiftScaleRotate(shift_limit = 0.0625,scale_limit = 0.1 ,rotate_limit = 3, p = 0.5),
                                      A.IAAAffine(scale = (1.0, 1.25), rotate = 0.0, p = 0.5)
                                      ],
                                        additional_targets={'image1':'image'})

        
    valid_transform = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((args.imagesize,args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    log = open(args.logfile, 'w')
    
    #K-Fold Cross Validation Starts here
    for fold in range(0, args.folds):
      print("\n***************************************************************************************\n FOLD: ", fold)
      print("\n***************************************************************************************\n FOLD: ", fold, file =log)
      
      #model = ResNet_18() #Will be pretrained on MS-Celeb
      best = 0
      
      #teach_pth = os.path.join(args.model_dir, str(fold)+' model_best.pth.tar')
      #teach_pth = '/content/drive/MyDrive/ijba_res18_naive.pth.tar'
      teach_path = '/content/drive/MyDrive/best_resnet_fer.pt'
      teacher = resnet18(False)
      teacher = load_base_model(teacher, teach_path)
      model = resnet18(False)
      #model = load_base_model(model)
      #model = RN18(BasicBlock, [2, 2, 2, 2], args.pre, device)
      classifier = Classifier(512, args.classes)
      print("=> reloading weights")
      teacher = nn.DataParallel(teacher).to(device)
      model = nn.DataParallel(model).to(device)
      model = load_base_model(model, '/content/drive/MyDrive/ijba_res18_naive.pth.tar')
      #classifier = load_class(classifier, args.pre)
      classifier = nn.DataParallel(classifier).to(device)

      #model = ResNet_no_hook()
      #model.to(device)
      """
      net = load_resnet18(False, True, '/content/drive/MyDrive/ijba_res18_naive.pth.tar', False, 7)
      net = nn.DataParallel(net).to(device)
      
      """
      optimizer =  torch.optim.Adam([{"params": model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}, 
                                 {"params": classifier.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
      lrs = []
      lrs.append(args.lr)
      """
      if args.resume:
          print("Resuming from previous fold")
          if os.path.isfile(args.resume):
              print("=> loading checkpoint '{}'".format(args.resume))
              checkpoint = torch.load(args.resume)
              ch = checkpoint['model_state_dict']
              
              net.load_state_dict(ch)
              
              optimizer.load_state_dict(checkpoint['optimizer'])
              print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
          else:
              print("=> no checkpoint found at '{}'".format(args.resume))  
      """
      
      #print(len(images))
      fileL = pd.read_csv((args.image_list), dtype='str', header = None)
      images= fileL.iloc[:, 0].values
      train_imgs, val_imgs = get_train_val_lists(images, args.folds, fold) 
      train_dataset = ImageList(args.root, args.subroot,  train_imgs, True,train_transform)
      tsne_dataset = ImageList(args.root, args.subroot,  train_imgs, False,valid_transform)
      val_dataset = ImageList(args.root, args.subroot, val_imgs, False, valid_transform)
      
      train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
      tsne_loader = torch.utils.data.DataLoader(tsne_dataset, args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True)
      val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=8)
      
      criterion = nn.CrossEntropyLoss().to(device)
      
      mse =nn.MSELoss().cuda()

      print("\nStarting Training\n")

      for epoch in range(args.start_epoch, args.epochs):
        if  epoch == 15 or epoch == 22:
          adjust_learning_rate(optimizer, epoch)
          lrs.append(optimizer.param_groups[0]["lr"])
          print(f'Updated lr: {lrs[-1]}\n', file = log)
          print(f'Updated lr: {lrs[-1]}\n')
        if epoch > 4:
          args.lamda = 100
        train(train_loader, model,  classifier, teacher, criterion, mse,  args.lamda, optimizer, epoch, args.epochs, log)
        acc, accnm, accm = validate(val_loader, model ,  classifier, epoch)
        #train(train_loader, net, criterion, optimizer, epoch, args.epochs, log)
        #acc = validate(val_loader, net, criterion, epoch)
        print("Epoch: {}   Validation Set Acc: {:.4f} Non-masked: {:.4f}  Masked: {:.4f}".format(epoch, acc, accnm, accm))
        print("Epoch: {}   Validation Set Acc: {:.4f}".format(epoch, acc), file = log)

        #Save best_acc and checkpoint
        is_best = acc > best
        best_acc = max(acc, best_acc)
        print('\n*********************************\nBest accuracy so far is : ', '%.4f'%best_acc)
        #print('\n*********************************\nBest accuracy so far is : ', '%.4f'%best_acc, log)

        if is_best: #Saving whenever model has learnt
            pth1, pth2 = save_checkpoint(model.state_dict(), classifier.state_dict(), 'checkpoint.pth.tar', fold)
            print("Saved")
            
      #Training Done!
      print("Training Done!")
      #model = load_base_model(model, pth1)
      #classifier = load_class(classifier, pth2)
      #Get hooked rN18 for GradCAM
      #net = resnet18CAM(pth1, pth2)
      #Grad CAM on val set
      #Grad_Cam(net, val_loader, '/content/drive/MyDrive/Student_Dataset_FER/Grad_Cam')
      #t-SNE plotss
      #T_SNE(model, classifier, tsne_loader, '/content/drive/MyDrive/Student_Dataset_FER/t_SNE', fold)
     
      
      #Confusion Matrix
      #conf_mat(model, classifier, val_loader, '/content/drive/MyDrive/Student_Dataset_FER/Cnf', fold)
      #For mix tsne
      #mix_pth = os.path.join(args.model_dir, str(fold)+' Mixmodel_best.pth.tar')
      #model = load_base_model(model, mix_pth)
      #T_SNE(model, classifier, tsne_loader, '/content/drive/MyDrive/Student_Dataset_FER/t_SNE_MIX', fold)

def save_checkpoint(msd, csd,  filename = 'checkpoint.pth.tar', fold = 0):
  
  full_bestname_model = os.path.join(args.model_dir, str(fold)+' KDrafmodel_best.pth.tar')
  full_bestname_class = os.path.join(args.model_dir, str(fold)+' KDrafclass_best.pth.tar')
  #if is_best:
  torch.save(msd, full_bestname_model)
  torch.save(csd, full_bestname_class)
  return full_bestname_model, full_bestname_class

def adjust_learning_rate(optimizer, epoch):
  for param_group in optimizer.param_groups:
        param_group['lr'] /= 10
    
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res
              

def train(train_loader, model,  classifier, teacher, criterion, mse, lamda, optimizer, epoch, n, log):
  model.train()
  classifier.train()
  teacher.eval()  #Important to set teacher model to eval
  running_loss = 0.0
  correct = 0
  total=0
  avgpool = nn.AdaptiveAvgPool2d(1)
  for batch_idx, (data1, data2, target, pth1, pth2) in enumerate(train_loader):
    data1 = data1.to(device)
    data2 = data2.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    out1 = teacher(data1) #Features from teacher
    out2 = model(data2) #Features from student model
    o1 = avgpool(out1)
    o1 = o1.squeeze(3).squeeze(2)
    o2 = avgpool(out2)
    o2 = o2.squeeze(3).squeeze(2)
   
    probs2 = classifier(o2) #Probs from student model
    
    loss1 = criterion(probs2, target) #Learn FER
    loss2 = mse(out1, out2) #Learn from teacher
    L = loss1 + lamda*loss2
    L.backward()
    optimizer.step()
    
    _, preds2 = torch.max(probs2, dim = 1)
    correct += torch.sum(preds2==target).item()
    
    total += target.size(0)
    acc = 100 * correct/total
    if batch_idx%args.print_freq == 0:
      print("Training Epoch: {}/{}\tLoss: {:.4f}\nTrain Accuracy: {:.4f}". format(epoch, n, L.item(), acc))
      print('Training Epoch: {}/{}\tLoss: {:.4f}\nTrain Accuracy: {:.4f}' . format(epoch, n, L.item(), acc), file = log)

def validate(val_loader, model, cls, epoch):
  model.eval()
  cls.eval()
  batch_loss = 0
  total=0
  correct=0
  c_nm = 0
  c_m = 0
  avgpool = nn.AdaptiveAvgPool2d(1)
  with torch.no_grad():
    for batch_idx, (data1, data2, target, pth1, pth2) in enumerate(val_loader):
      data1 = data1.to(device)
      data2 = data2.to(device)
      target = target.to(device)
      out1 = model(data1)
      out2 = model(data2)
      o1 = avgpool(out1)
      o1 = o1.squeeze(3).squeeze(2)
      o2 = avgpool(out2)
      o2 = o2.squeeze(3).squeeze(2)
      probs1 = cls(o1)
      probs2 = cls(o2)
      _, preds1 = torch.max(probs1, dim = 1)
      _, preds2 = torch.max(probs2, dim = 1)
      correct += torch.sum(preds1==target).item()
      c_nm += torch.sum(preds1==target).item()
      c_m += torch.sum(preds2==target).item()
      correct += torch.sum(preds2==target).item()
      total += target.size(0)*2
  acc = 100 * correct/total
  acc_nm = 100 * c_nm/(total/2)
  acc_m = 100 * c_m/(total/2)
  #print(f"Validation Set Accuracy: {(100 * correct/total):.4f}\n")
  return acc, acc_nm, acc_m


if __name__ == "__main__":
    
    main(args)
    print("Completed K-fold Cross Validation!")  
