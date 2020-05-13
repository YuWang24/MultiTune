# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:04:57 2020

@author: s1013_000
"""
import shutil, os, glob

def makeFolder(srcDir):
    if not os.path.isdir(srcDir):
        os.mkdir(srcDir)
    else:
        shutil.rmtree(srcDir)
        os.mkdir(srcDir)
    subfolder = ['train', 'val', 'test']
    for sub in subfolder:
        os.mkdir(srcDir+'/'+sub)
        if sub != 'test':
            for i in range(1,101):
                if i<10:
                    os.mkdir(srcDir+'/'+sub+'/'+'000'+str(i))
                elif i>=10 and i<=99:
                    os.mkdir(srcDir+'/'+sub+'/'+'00'+str(i))
                else:
                    os.mkdir(srcDir+'/'+sub+'/'+'0'+str(i))

def moveAllFilesinDir(srcDir, dstDir, N):
    # Check if both the are directories
    subfolder = ['train', 'val', 'test']
    for sub in subfolder:
        if os.path.isdir(os.path.join(srcDir, sub)) and os.path.isdir(os.path.join(dstDir, sub)) :
            if sub != 'test':
                for i in range(1,101):
                    if i<10:
                        source = srcDir+'/'+sub+'/'+'000'+str(i)
                        des    = dstDir+'/'+sub+'/'+'000'+str(i)
                    elif i>=10 and i<=99:
                        source = srcDir+'/'+sub+'/'+'00'+str(i)
                        des    = dstDir+'/'+sub+'/'+'00'+str(i)
                    else:
                        source = srcDir+'/'+sub+'/'+'0'+str(i)
                        des    = dstDir+'/'+sub+'/'+'0'+str(i)
                    count = 0
                # Iterate over all the files in source directory
                    files = os.listdir(source)
                    for f in files:
                        count+=1
                        if count>N:
                            break
                        else:
                            shutil.copy2(source+'/'+f, des)
#                     print(count)
            else:
                files = os.listdir(srcDir+'/'+sub)
                for f in files:
                    shutil.copy2(srcDir+'/'+sub+'/'+f, dstDir + sub + '/');
        else:
            print("srcDir & dstDir should be Directories")