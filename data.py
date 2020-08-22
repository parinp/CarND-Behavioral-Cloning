from zipfile import ZipFile

filename1 = '/home/workspace/own_data1.zip'

with ZipFile(filename1, 'r') as zip:
    zip.extractall('/home/workspace/data/')
    
filename2 = '/home/workspace/own_data2.zip'

with ZipFile(filename2, 'r') as zip:
    zip.extractall('/home/workspace/data/')

filename3 = '/home/workspace/own_data3.zip'

with ZipFile(filename3, 'r') as zip:
    zip.extractall('/home/workspace/data/')
    
filename4 = '/home/workspace/own_data4.zip'

with ZipFile(filename4, 'r') as zip:
    zip.extractall('/home/workspace/data/')
    
filename5 = '/home/workspace/data.zip'

with ZipFile(filename5, 'r') as zip:
    zip.extractall('/home/workspace/data/')