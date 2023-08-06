import re
import anarci
from ast import literal_eval

import numpy as np
import pandas as pd
import torch


def get_anarci_pos(batch):
    anarci_numbering = anarci.run_anarci(batch, scheme='imgt')[1][0][0][0]

    index = []
    # ignore "-"
    for element in anarci_numbering:
        idx = str(element[0][0]) + element[0][1]  # (1, 'A') or (1, ' ')
        aa = element[1]  # 'Q' or '-'
        if aa != '-':
            index.append(idx)
    try:
        new_index = list(map(lambda id: int(id), index))
    except:
        new_index = []
        for id in index:
            try:
                new_index.append(int(id))
            except:
                pos_map = {
                    '111A': 129, '111B': 130, '111C': 131, '111D': 132, '111E': 133, 
                    '112A': 139, '112B': 138, '112C': 137, '112D': 136, '112E': 135, '112F': 134, 
                }
                if id not in pos_map.keys() and int(re.sub('[a-zA-Z]','',id)) < 111:
                    new_index.append(int(re.sub('[a-zA-Z]','',id)))
                elif id in pos_map.keys():
                    new_index.append(pos_map[id])
                elif int(id[:3]) == 111:
                    new_index.append(133)
                elif int(id[:3]) == 112:
                    new_index.append(134)
        
    new_index = [0] + new_index + [140] * (168 - 1 - len(new_index))
    if len(new_index) > 168:
        new_index = new_index[:168]
        new_index[-1] = 140
    return {'position_ids': torch.tensor(new_index)}

if __name__=="__main__":
    sequence = "AVQLQESGGGLVQAGGSLRLSCTVSARTSSSHDMGWFRQAPGKEREFVAAISWSGGTTNYVDSVKGRFDISKDNAKNAVYLQMNSLKPEDTAVYYCAAKWRPLRYSDNPSNSDYNYWGQGTQVTVSS"
    print(get_anarci_pos(sequence))
    
    

