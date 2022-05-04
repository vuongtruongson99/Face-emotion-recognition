#encoding:utf-8
import h5py
import os
import sys
class HDF5DatasetWriter:
    def __init__(self,dims,outputPath,dataKey='images',bufSize=1000):
        if os.path.isfile(outputPath):
            print("The supplied ‘outputPath‘ already "
                "exists and cannot be overwritten. Manually delete "
                "the file before continuing.", outputPath)
            os.system('rm -rf %s'%outputPath)

        self.db = h5py.File(outputPath,'w')
       
        self.data = self.db.create_dataset(dataKey,dims,dtype='float')
        self.labels = self.db.create_dataset('labels',(dims[0],),dtype='int')
        # 缓冲大小
        self.bufSize = bufSize
        self.buffer = {"data":[],"labels":[]}
        # 索引
        self.idx = 0

    def add(self,rows,labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        if len(self.buffer['data']) >= self.bufSize:
            self.flush()

    def flush(self):
        # 将buffers写入磁盘并初始化buffer
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i # 指针
        self.buffer = {"data":[],"labels":[]}

    def storeClassLabels(self,classLabels):
        # 一个dataset存储数据标签名称
        # dt = h5py.special_dtype(vlen = unicode)  # python2.7
        dt = h5py.special_dtype(vlen = str)        # python3
        labelSet =self.db.create_dataset('label_names',(len(classLabels),),dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer['data']) >0 :
            self.flush()
        # 关闭dataset
        self.db.close()
