#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from bs4.element import Tag
import urllib


# In[27]:


driver = webdriver.Chrome("chromedriver")
driver.get('https://richmondin.craigslist.org/search/boo?')
time.sleep(3)


# In[29]:


soup = BeautifulSoup(driver.page_source,'lxml')
rowArray = soup.find_all("li", { "class":"result-row"})

for row in rowArray:
    img = row.find("img")
    if img is None:
        continue
    urllib.request.urlretrieve(img['src'],img['src'][-20:] )   

