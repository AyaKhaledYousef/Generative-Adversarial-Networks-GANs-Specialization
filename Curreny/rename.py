import os
import glob
count=0
for i in glob.glob('NEW/200 pounds/*.JPG'):
    print(i)
    count+=1
    os.rename(i,'NEW/200 pounds/200-{}'.format(count))
