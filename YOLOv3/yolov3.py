import torch
import torch.nn as nn
from util import *

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_cfg(file):
    """
    Takes a config file and returns block in the neural network to be built. 
    
    """
    with open(file, 'r') as file:
        lines = []
        for line in iter(lambda: file.readline(), ''):   
            if line[0] != '#':
                line = line.rstrip().lstrip()
                if len(line)>0:
                    lines.append(line)
    
    blocks = []
    block = {}
    
    for line in lines:
        if line[0] == "[":               
            if len(block) != 0:          
                blocks.append(block)    
                block = {}            
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, block in enumerate(blocks[1:]):

        module = nn.Sequential()

        if block["type"] == "convolutional":

            try: 
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True


            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0


            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)

            activation = block["activation"]
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("batch_norm_{0}".format(index), activn)

        elif block["type"] == "upsample":

            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)

        elif block["type"] == "route":

            block["layers"] = block["layers"].split(',')

            start = int(block["layers"][0])

            try:
                end = int(block["layers"][1])
            except:
                end = 0


            if start > 0: start = start - index
            if end > 0: end = end - index

            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]



        elif block["type"] == "shortcut":

            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif block["type"] == "yolo":

            mask = [int(x) for x in block["mask"].split(',')]

            anchors = list(map(int,blocks[-1]["anchors"].split(',')))
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return net_info, module_list

class YOLOv3(nn.Module):
    def __init__(self, cfgfile):
        super(YOLOv3, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, isCUDA):
        if isCUDA: x = x.to("cuda")
        modules = self.blocks[1:]
        outputs = {}
        
        write = 0
        for i, module in enumerate(modules):
            
            module_type = module["type"]
            
            if module_type in {"convolutional", "upsample"}:
                x = self.module_list[i](x)
                
            elif module_type == "route":
                
                layers = module["layers"]
                layers = [int(l) for l in layers]
                
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, isCUDA)
                
                if not write:
                    detections = x
                    write = 1
                
                else:
                    detections = torch.cat((detections, x), 1)
                
                
            outputs[i] = x
            
        return detections



