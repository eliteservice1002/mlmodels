#!/usr/bin/env python
# coding: utf-8

# In[4]:


import json
import time

import numpy as np

import luigi

# In[5]:


class Split_text(luigi.Task):
    filename = luigi.Parameter()
    split_size = luigi.IntParameter(default=5)

    def output(self):
        return [luigi.LocalTarget("text-%d.txt" % (i)) for i in range(self.split_size)]

    def run(self):
        with open(self.filename) as fopen:
            texts = list(filter(None, fopen.read().split("\n")))
        splitted_list = np.array_split(texts, self.split_size)
        for i in range(len(splitted_list)):
            splitted_list[i] = splitted_list[i].tolist()
            time.sleep(2)
        for no, file in enumerate(self.output()):
            with file.open("w") as fopen:
                fopen.write("\n".join(splitted_list[no]))


class WordCount(luigi.Task):
    filename = luigi.Parameter()
    split_size = luigi.IntParameter(default=5)

    def requires(self):
        return Split_text(filename=self.filename, split_size=self.split_size)

    def output(self):
        return luigi.LocalTarget("wordcount.txt")

    def run(self):
        count, texts = {}, []
        for file in self.input():
            with file.open("r") as fopen:
                texts.append(list(filter(None, fopen.read().split("\n"))))
        texts = " ".join(sum(texts, []))
        texts = texts.split()
        for word in texts:
            count[word] = count.get(word, 0) + 1
        with self.output().open("w") as fopen:
            fopen.write(json.dumps(count))


# In[6]:


luigi.build(
    [WordCount(filename="big-text.txt", split_size=10)],
    scheduler_host="localhost",
    scheduler_port=8082,
)


# In[ ]:
