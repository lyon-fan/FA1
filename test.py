def str_to_hex(s):
    return ' '.join([hex(ord(c)).replace('0x', '') for c in s])

def hex_to_str(s):
    return ''.join([chr(i) for i in [int(b, 16) for b in s.split(' ')]])

a = "010100000000000000010"
b = str_to_hex(a)
c=  hex_to_str(b)


def hex_to_str1(b):
    s = ''
    for i in b:
        s += '{0:0>2}'.format(str(hex(i))[2:])
    return(s)


s = "01110101 01110011 01100101 00100000 01110111 01100101 01100100 01101110 01100101 01110011 01100100 01100001 01111001 00100000 01100110 01101111 01110010 00100000 01110100 01101000 01100101 00100000 01100001 01101110 01110011 01110111 01100101 01110010"
bit = s.split()
dec = map(lambda x:int(x,2),bit)
print(''.join(map(chr,dec)))