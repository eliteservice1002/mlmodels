#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
from urllib.request import urlopen, urlretrieve

from bs4 import BeautifulSoup
from tqdm import tqdm

# In[2]:


prefix = "https://tspace.library.utoronto.ca"
save_dir = "./data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

base_url = "https://tspace.library.utoronto.ca/handle/1807/24"
urls = [base_url + str(i) for i in range(488, 502)]
for url in urls:
    soup = BeautifulSoup(urlopen(url).read(), "html5lib")
    targets = soup.findAll("a", href=re.compile(r"/bitstream/.*.wav"))

    for a in tqdm(targets, total=len(targets), ncols=70):
        link = a["href"]

        audio_save_loc = save_dir + link.split("/")[-1]
        if os.path.isfile(audio_save_loc):
            print("File Already Exists")
        urlretrieve(prefix + a["href"], audio_save_loc)

        with open(audio_save_loc.replace(".wav", ".txt"), "w") as f:
            f.write("say the word " + link.split("_")[-2])


# In[ ]:
