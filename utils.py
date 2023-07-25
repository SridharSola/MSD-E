from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
# Skleran
from sklearn.datasets import load_digits 
from sklearn.manifold import TSNE # for t-SNE dimensionality reduction


def conf_mat(model, cls, test, Cfile, fold):
  y_pred1 = []
  y_pred2 = []
  y_true = []
  model.eval()
  cls.eval()
  avgpool = nn.AdaptiveAvgPool2d(1)
  # iterate over test data
  for inputs1, inputs2, labels, p, q in test:
          out1 = model(inputs1) # Feed Network
          o1 = avgpool(out1)
          o1 = o1.squeeze(3).squeeze(2)
          
          output1 = cls(o1)
          out2= model(inputs2) # Feed Network
          o2 = avgpool(out2)
          o2 = o2.squeeze(3).squeeze(2)
          
          output2 = cls(o2)
          #output1 = torch.cat((output1, output2), dim=0)
          #output = output1
          output1 = (torch.max(torch.exp(output1), 1)[1]).data.cpu().numpy()
          y_pred1.extend(output1) # Save Prediction
          output2 = (torch.max(torch.exp(output2), 1)[1]).data.cpu().numpy()
          y_pred2.extend(output1) # Save Prediction
          #labels = torch.cat((labels, labels), dim=0)
          labels = labels.data.cpu().numpy()
          y_true.extend(labels) # Save Truth

  # constant for classes
  classes = ('Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral')

  # Build confusion matrix
  cf_matrix1 = confusion_matrix(y_true, y_pred1)
  df_cm1 = pd.DataFrame(cf_matrix1/np.sum(cf_matrix1) *10, index = [i for i in classes],
                      columns = [i for i in classes])

  df_conf_norm = df_cm1 / df_cm1.sum(axis=1)
  plt.figure(figsize = (12,7))
  sn.heatmap(df_conf_norm, annot=True)
  p = str(fold) + '1.png'
  pth = os.path.join(Cfile,p)
  plt.savefig(pth)
  cf_matrix2 = confusion_matrix(y_true, y_pred2)
  df_cm2 = pd.DataFrame(cf_matrix2/np.sum(cf_matrix2) *10, index = [i for i in classes],
                      columns = [i for i in classes])

  df_conf_norm = df_cm2 / df_cm2.sum(axis=1)
  plt.figure(figsize = (12,7))
  sn.heatmap(df_conf_norm, annot=True)
  p = str(fold) + '2.png'
  pth = os.path.join(Cfile,p)
  plt.savefig(pth)

def Grad_Cam(net, loader, Gfile):
  net.eval()
  net.to(device)
  for j, (im, img, l, pt, pth) in enumerate(loader):
    img = img.to(device)
    pred1 = net(img)
    print(pred1)
    indx = pred1.argmax(dim = 1) # 1*1
    #print(indx)
    # get the gradient of the output with respect to the parameters of the model
    pred1[:, indx].backward()
    # pull the gradients out of the model
    gradients = net.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations of the last convolutional layer
    activations = net.get_activations(img).detach()
    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    #Superimposing
    img = cv2.imread(pth[0])
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    #cv2.imwrite('./map.jpg', superimposed_img)
    #if j%100:
       # cv2_imshow(superimposed_img)
    # draw the heatmap
    #plt.matshow(heatmap.squeeze())
    pth = pth[0]
    #print(pth)
    end = pth.split('/')[-1]
    end = str(j) + 'pred' + str(indx.item()) + 'label' + str(l.item()) + '.jpg'
    cv2.imwrite(os.path.join(Gfile, end), superimposed_img)



def scale_to_01_range(x):

    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range


def test( model1, cls, test_loader):
    
    model1.eval()  # Change model to 'eval' mode.
    cls.eval()
   
    correct = 0
    correct1 = 0
    correct2  = 0
    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0
    correct  = 0
    avgpool = nn.AdaptiveAvgPool2d(1)
    with torch.no_grad():
        for i,(images1,images2,labels, p1,p2) in enumerate(test_loader):
            images1 = (images1).to(device)
            images2 = (images2).to(device)
            out1= model1(images1)
            out2= model1(images2)
            o1 = avgpool(out1)
            features1 = o1.squeeze(3).squeeze(2)
            o2 = avgpool(out2)
            features2 = o2.squeeze(3).squeeze(2)
            outputs1 =cls(features1)
            outputs2 =cls(features2)
            _, pred1 = torch.max(outputs1.data, 1)
            _, pred2 = torch.max(outputs2.data, 1)
            total1 += labels.size(0)*2
            correct1 += (pred1.cpu() == labels).sum()

            
            #outputs1 = torch.cat((outputs1, outputs2), dim=0)
            
            #features1 = torch.cat((features1, features2), dim=0)
            #features = features1
            avg_output1 = outputs1.data 
            avg_output2 = outputs2.data
            #labels = torch.cat((labels, labels), dim=0)
            _, avg_pred1 = torch.max(avg_output1, 1)
            _, avg_pred2 = torch.max(avg_output2, 1)
            #correct += (avg_pred.cpu() == labels).sum()
                  
            #print('\n',i, preds_eval, target, correct)
            if i == 0:
          
              all_predicted1 = avg_pred1
              all_predicted2 = avg_pred2
              all_targets = labels
            else:
               
              all_predicted1 = torch.cat((all_predicted1, avg_pred1), 0)
              all_predicted2 = torch.cat((all_predicted2, avg_pred2), 0)
              all_targets = torch.cat((all_targets, labels), 0)

    
    
            #image_paths += batch['image_path']

            current_outputs1 = features1.cpu().numpy()
            current_outputs2 = features2.cpu().numpy()
            if i == 0:
          
              features_outputs1 = current_outputs1
              features_outputs2 = current_outputs2
            else:
              features_outputs1 = np.concatenate((features_outputs1, current_outputs1))
              features_outputs2 = np.concatenate((features_outputs2, current_outputs2))
    
    #acc = (float(correct) / len(test_loader.dataset) )
         
    #print(' Accuracy: ',acc)
    
    return all_targets, all_predicted1, features_outputs1, all_predicted2, features_outputs2

def T_SNE(model1, cls, loader, Tfile, fold):
    all_targets, all_predicted1, features_outputs1, all_predicted2, features_outputs2 = test( model1, cls, loader)
    
    tsne1 = TSNE(n_components=2).fit_transform(features_outputs1)
    tsne2 = TSNE(n_components=2).fit_transform(features_outputs2)
    
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne1[:, 0]
    ty = tsne1[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    all_predicted = all_predicted1.cpu().numpy()
    all_targets = all_targets.cpu().numpy()
    #print(tx.shape,ty.shape)
    #palette = np.array(sb.color_palette("hls", 10))  #Choosing color palette 
    # for every class, we'll add a scatter plot separately

    classes = ('Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral')

    plot = ax.scatter(tx, ty, c = all_targets, s=5,cmap="nipy_spectral")

    plt.legend(handles=plot.legend_elements()[0], labels=classes)

    p = str(fold) + '1.png'
    pth = os.path.join(Tfile, p)
    plt.savefig(pth)

    tx = tsne2[:, 0]
    ty = tsne2[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    all_predicted = all_predicted2.cpu().numpy()
    #all_targets = all_targets.cpu().numpy()
    #print(tx.shape,ty.shape)
    #palette = np.array(sb.color_palette("hls", 10))  #Choosing color palette 
    # for every class, we'll add a scatter plot separately

    classes = ('Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral')

    plot = ax.scatter(tx, ty, c = all_targets, s=5,cmap="nipy_spectral")

    plt.legend(handles=plot.legend_elements()[0], labels=classes)

    p = str(fold) + '2.png'
    pth = os.path.join(Tfile, p)
    plt.savefig(pth)




