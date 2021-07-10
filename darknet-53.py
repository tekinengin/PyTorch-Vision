
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