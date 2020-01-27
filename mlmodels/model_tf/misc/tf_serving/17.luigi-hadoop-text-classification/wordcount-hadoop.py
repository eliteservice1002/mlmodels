#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import time

import numpy as np

import luigi
import luigi.contrib.hadoop
import luigi.contrib.hdfs

# In[2]:


class Split_text(luigi.Task):
    filename = luigi.Parameter()
    split_size = luigi.IntParameter(default=5)

    def output(self):
        return [
            luigi.contrib.hdfs.target.HdfsTarget("/user/input_text/text-%d.txt" % (i))
            for i in range(self.split_size)
        ]

    def run(self):
        with open(self.filename) as fopen:
            texts = list(filter(None, fopen.read().split("\n")))
        splitted_list = np.array_split(texts, self.split_size)
        for i in range(len(splitted_list)):
            splitted_list[i] = splitted_list[i].tolist()
        for no, file in enumerate(self.output()):
            with file.open("w") as fopen:
                fopen.write("\n".join(splitted_list[no]))


class WordCount_Hadoop(luigi.contrib.hadoop.JobTask):
    filename = luigi.Parameter()
    split_size = luigi.IntParameter(default=5)

    def requires(self):
        return Split_text(filename=self.filename, split_size=self.split_size)

    def output(self):
        return luigi.contrib.hdfs.target.HdfsTarget(
            "/user/input_text/count.txt", format=luigi.contrib.hdfs.PlainDir
        )

    def mapper(self, line):
        sentences = list(filter(None, line.split("\n")))
        for sentence in sentences:
            for word in sentence.split():
                yield word, 1

    def reducer(self, key, values):
        yield key, sum(values)


# In[3]:


if __name__ == "__main__":
    luigi.build(
        [WordCount_Hadoop(filename="big-text.txt", split_size=10)],
        scheduler_host="localhost",
        scheduler_port=8082,
    )


# In[ ]:
