# coding: utf-8
import sys
import os.path

file = open(sys.argv[1], 'r').readlines()
basename = os.path.basename(sys.argv[1])
name = os.path.splitext(basename)
output = 'Predataset/out_{}.csv'.format(name[0])

file2=[]
for line in file:
    #extract attrivute
    if (line.find('@inputs') != -1):
        attribute = line.replace('@inputs', '')
        attribute = attribute.replace(' ', '')
        attribute = attribute.strip()
        attribute += ',Label\n'
       # print(attribute)
    elif (line.find('@input') != -1):
        attribute = line.replace('@input', '')
        attribute = attribute.replace(' ', '')
        attribute = attribute.strip()
        attribute += ',Label\n'

    #delete strings
    if (line.find('@') != -1):
        del line
    #replace class label
    #positive=1, negative=0
    else:
        if (line.find('positive') != -1):
            pos = line.replace('positive', '1')
            file2.append(pos)
        elif (line.find('negative') != -1):
            neg = line.replace('negative', '0')
            file2.append(neg)
file2.insert(0, attribute)


out_file = open(output, 'w')
for i in range(len(file2)):
    out_file.write(file2[i])
out_file.close()

