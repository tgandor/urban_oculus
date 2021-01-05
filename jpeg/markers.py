

names = {
    0: 'FF00',
    0xc4: 'DHT',
    0xd8: 'SOI',
    0xd9: 'EOI',
    0xda: 'SOS',
    0xdb: 'DQT',
}
markers = {}  # reverse names
for d, m in names.items():
    markers[m] = d

# APP_x
for i, d in enumerate(range(0xe0, 0xef+1)):
    m = 'APP{}'.format(i)
    names[d] = m
    markers[m] = d

standalone = {
    0, # not a marker
    0xd8, # SOI
    0x01, # TEM
    0xd9, # EOI (but nothing after)
    *range(0xd0, 0xd7+1), # RST
}

# SOF_x - Start of Frame x
not_sof = {
    0xc4, # DHT Define Huffman Table
    0xc8, # JPG JPEG Extensions
    0xcc, # DAC Define Arithmetic Coding
}
for i, d in enumerate(range(0xc0, 0xcf+1)):
    if d in not_sof:
        continue
    m = 'SOF{}'.format(i)
    names[d] = m
    markers[m] = d

# RST_x - Restart Marker x
for i, d in enumerate(range(0xd0, 0xd7+1)):
    standalone.add(d)
    m = 'RST{}'.format(i)
    names[d] = m
    markers[m] = d
