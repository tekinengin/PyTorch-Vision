import torch
import torch.nn as nn

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

            layers = block["layers"].split(',')

            start = int(layers[0])

            try:
                end = int(layers[1])
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