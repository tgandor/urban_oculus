from . import markers

'''
References:
https://stackoverflow.com/questions/1557071/determining-the-size-of-a-jpeg-jfif-image#1602428
https://www.impulseadventure.com/photo/jpeg-decoder.html
'''

def get_QTs(filename, with_id=False):
    results = []
    offset = 1 - int(with_id)
    with open(filename, 'rb') as f:
        c = ord(f.read(1))
        d = ord(f.read(1))
        assert c == 0xff
        assert d == markers.markers['SOI']
        while True:
            c = ord(f.read(1))
            if c == 0xff:
                d = ord(f.read(1))
                if d in markers.standalone:
                    continue
                h = ord(f.read(1))
                l = ord(f.read(1))
                size = 256 * h + l
                data = f.read(size - 2)  # 2 bytes already there
                name = markers.names[d]
                if name == 'DQT':
                    if size == 67: # single QT
                        results.append(data[offset:].hex())
                    elif size == 132: # double QT
                        results.append(data[offset:65].hex())
                        results.append(data[offset+65:].hex())
                    else:
                        raise ValueError('Malformed DQT')
                if name == 'SOS':
                    break
    return results
