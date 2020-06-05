#!/usr/bin/env python

import torch

count = torch.cuda.device_count()
print('Device count:', count)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

for i in range(count):
    #Additional Info when using cuda
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(i))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(i)/1024**3, 1), 'GB')
