import torch
import numpy as np

def predict_transform(prediction, ind_dim, anchors, num_classes, isCUDA = True):
    
    batch_size = prediction.size(0)
    stride = ind_dim // prediction.size(2)
    grid_size = ind_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    anchors = [(anc[0]/stride, anc[1]/stride) for anc in anchors]
    
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if isCUDA:
        
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        
        
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset 
        
    anchors = torch.FloatTensor(anchors)
    
    if isCUDA:
        anchors = anchors.cuda()
        
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid(prediction[:,:,5: 5 + num_classes])
    
    prediction[:,:,:4] *= stride
    
    return prediction