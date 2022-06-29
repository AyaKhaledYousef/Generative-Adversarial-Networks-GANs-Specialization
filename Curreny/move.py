import os
import glob
import shutil

for i in glob.glob('NEW/*/*'):
    shutil.move(i,'images/')

