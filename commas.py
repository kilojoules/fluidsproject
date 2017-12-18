s=[] 
for d in open('d', 'r').readlines(): 
   s.append(d.strip('\n'))
print ','.join(s)
