#!/usr/bin/env python
# coding: utf-8

# In[1]:


# synthetic classification dataset
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# create scatter plot for samples from each class
for class_value in range(2):
 # get row indexes for samples with this class
 row_ix = where(y == class_value)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show() 
#อิมพอร์ตไลบารี ทำการ classification โดยสุ่มมา 1000 ตัวที่ไม่ซ้ำซ้อน โดยพลอตกราฟแบบ scatter และทำการ show plot ออกมา


# In[2]:


# affinity propagation clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = AffinityPropagation(damping=0.9)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอร์ตไลบารี ทำการ classification โดยสุ่มมา 1000 ตัวที่ไม่ซ้ำซ้อนกำหนดโมเดลและทำการ train model
#ทำการแบ่งกลุ่มแบบ unique และแบ่งกรุ๊ป โดยพลอตกราฟแบบ scatter ทำการ show plot ออกมา


# In[3]:


# agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอร์ตไลบารี ทำการ classification โดยสุ่มมา 1000 ตัวที่ไม่ซ้ำซ้อนกำหนดโมเดลและทำการ train model
#กำหนดโมเดลโดยใช้ Agglomerative model ทำการ Train model และแยก unique และวนลูปหาข้อมูลที่จับกลุ่มกันอีกที
#คำนวณหาค่าความใกล้ชิด cluster ที่อยู่ใกล้กันจะถูกจับรวมตัวกัน และจะวนทำเช่นนี้ไปเรื่อย ๆ จนกว่าจะกลายเป็น cluster เดียวในที่สุด


# In[4]:


# birch clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = Birch(threshold=0.01, n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอร์ตไลบารี ทำการ classification โดยสุ่มมา 1000 ตัวที่ไม่ซ้ำซ้อนกำหนดโมเดลและทำการ train model
#กำหนดโมเดลโดยใช้ birch model ทำการ Train model และแยก unique และวนลูปหาข้อมูลที่จับกลุ่มกันอีกที
#เหมาะกับข้อมูลใหญ่ๆ โดยจะจัดทกับำข้อมุลโดยย่อที่เราทำการสรุปมาไว้แล้ว


# In[5]:


# dbscan clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = DBSCAN(eps=0.30, min_samples=9)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอร์ตไลบารี ทำการ classification โดยสุ่มมา 1000 ตัวที่ไม่ซ้ำซ้อนกำหนดโมเดลและทำการ train model
#กำหนดโมเดลโดยใช้ DBSCAN model ทำการ Train model และแยก unique และวนลูปหาข้อมูลที่จับกลุ่มกันอีกที
#เป็นการหาบริเวณที่ข้อมูลเกาะกลุ่มกัน ซึ่งสามารถคำนวณได้จาก data point ที่อยู่รอบๆ ในรัศมีที่กำหนด


# In[6]:


# k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอรตไลบารีของ K-Means ทำการ classification กำหนดโมเดลและ train นำข้อมูลมาทำนาย
#หาค่า unique และจัดกลุ่มแต่เราวนลูป cluster เพื่อหาค่า unique อีกที และทำการจัดกลุ่มและโชวืออกมาเป็น scatter
#K-means จะจัดหลุ่มดดยเรากำหนดจำนวณ clustering ไว้ 2 กลุ่ม


# In[7]:


# mini-batch k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = MiniBatchKMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอรตไลบรารี่ minibatchkmeans และทำการ classification เรากำหนดค่า data ดดยเราใช้ แค่ X
#กำหนดค่า model และทำการ train เพื่อพรีดิกและหาค่า unique ทำการวนลูปเพื่อหาค่าจาก cluster ที่แตกต่างกัน และนำมา scatter plot
#Mini-Batch K- หมายถึงการจัดกลุ่ม เป็นตัวแปรในการจัดกลุ่ม K-mean โดยที่ขนาดของชุดข้อมูลที่กำลังพิจารณาถูกจำกัดไว้ 
#การทำคลัสเตอร์ K-mean แบบปกติจะดำเนินการกับชุดข้อมูลทั้งหมด/ชุดพร้อมกัน 
#ในขณะที่การทำคลัสเตอร์ K-mean แบบ Mini-batch แบ่งชุดข้อมูลออกเป็นส่วนย่อย
#ชุดย่อยจะถูกสุ่มตัวอย่างจากชุดข้อมูลทั้งหมด และสำหรับการวนซ้ำใหม่แต่ละครั้ง จะมีการเลือกตัวอย่างสุ่มใหม่และใช้เพื่ออัปเดตตำแหน่งของเซนทรอยด์


# In[8]:


# mean shift clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import MeanShift
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = MeanShift()
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#ทำการอิมพอรตไลบรารีและ classification กำหนดค่าโมเดลและการทำนาย กำหนดกลุ่มโดยใช้ unique และวนลูปเพื่อแยกกรุ๊ปและ scatter plot เอาค่าออกมา
#Mean shift เป็น density-based clustering ส่วน k-means เป็น centroid-based clustering 
#กล่าวคือ Mean shift จะใช้คอนเซปของ kernel density estimation (KDE) ในการหาตัวแทนกลุ่มของข้อมูล หรือ centroid


# In[9]:


# optics clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import OPTICS
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอรตไลบรารี่ optics และทำการ classification เรากำหนดค่า data ดดยเราใช้ แค่ X
#กำหนดค่า model และทำการ train เพื่อพรีดิกและหาค่า unique ทำการวนลูปเพื่อหาค่าจาก cluster ที่แตกต่างกัน และนำมา scatter plot
#เป็นอัลกอริทึมที่ต่อยอดมาจาก DBSCAN โดยมีหลักความเข้าใจคล้ายกันคือ การหาบริเวณที่ข้อมูลเกาะกลุ่มกันโดยมีรัศมีค่า ε (เอปไซลอน) คล้ายกับ DBSCAN 
#แต่จะมีตัวแปรเพิ่มคือ Core Distance และ Reachability Distance โดย Core Distance จะเป็นค่าต่ำสุดของรัศมีเพื่อจัดกลุ่มจุดข้อมูล
#และ Reachability Distance คือค่าระยะห่างระหว่างจุดข้อมูล 2 จุดโดยมีจุด p ซึ่งเป็นศูนย์กลางของ Core Distance เป็นตัวตั้งอัลกอริทึมนี้
#จะช่วยวิเคราะห์ความหนาแน่นได้หลายมิติ มากกว่าที่ DBSCAN ทำได้


# In[10]:


# spectral clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = SpectralClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอรตไลบรารี่ spectral clustering และทำการ classification เรากำหนดค่า data ดดยเราใช้ แค่ X
#กำหนดค่า model และทำการ train เพื่อพรีดิกและหาค่า unique ทำการวนลูปเพื่อหาค่าจาก cluster ที่แตกต่างกัน และนำมา scatter plot
#ใช้เทคนิคการจัดกลุ่มสเปกตรัมใช้สเปกตรัมของเมทริกซ์ความคล้ายคลึงกันของข้อมูลเพื่อลดมิติข้อมูลก่อนที่จะจัดกลุ่มในมิติข้อมูลที่น้อยลง


# In[11]:


# gaussian mixture clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
n_clusters_per_class=1, random_state=4)
# define the model
model = GaussianMixture(n_components=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
 # get row indexes for samples with this cluster
 row_ix = where(yhat == cluster)
 # create scatter of these samples
 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
#อิมพอรตไลบรารี่ gaussian mixture clustering และทำการ classification เรากำหนดค่า data ดดยเราใช้ แค่ X
#กำหนดค่า model และทำการ train เพื่อพรีดิกและหาค่า unique ทำการวนลูปเพื่อหาค่าจาก cluster ที่แตกต่างกัน และนำมา scatter plot
#เราจะดุจากจุดพีคของแต่ละจุด มองภาพคล้ายภาพระฆังคว่ำแล้วเอาจุดพีคที่เป็นจุดกึ่งกลางที่ซ้อนๆจะเห็นว่าตัวแปรตัวเดียวกันนี้มันมีตัวแปรแฝงอยู่ มัน mix กันอยู่ 


# In[ ]:


K-means vs 1 Algorithm (เชิงลึก)
K-means ไม่มีคำตอบตายตัว โดยหน้าที่หลักของ K-means คือการแบ่งกลุ่ม แบบ Clustering 
ซึ่งการแบ่งกลุ่มในลักษณะนี้จะใช้พื้นฐานทางสถิติ ซึ่งหน้าที่ของ clustering คือการจับกลุ่มของข้อมูลที่มีลักษณะใกล้เคียงกันเป็นกลุ่มเดียวกัน
k คือตัวเลขจำนวนเต็มที่ระบุจำนวน segments ที่ต้องการจากข้อมูลชุดนั้น 
ส่วนคำว่า means บอกเราว่าข้อมูลที่จะใช้กับ algorithm นี้ได้ต้องเป็นตัวเลขที่หา “ค่าเฉลี่ย” ได้เท่านั้น

optics clustering
เป็นอัลกอริทึมที่ต่อยอดมาจาก DBSCAN โดยมีหลักความเข้าใจคล้ายกันคือ การหาบริเวณที่ข้อมูลเกาะกลุ่มกันโดยมีรัศมีค่า ε (เอปไซลอน) คล้ายกับ DBSCAN 
แต่จะมีตัวแปรเพิ่มคือ Core Distance และ Reachability Distance โดย Core Distance จะเป็นค่าต่ำสุดของรัศมีเพื่อจัดกลุ่มจุดข้อมูล
และ Reachability Distance คือค่าระยะห่างระหว่างจุดข้อมูล 2 จุดโดยมีจุด p ซึ่งเป็นศูนย์กลางของ Core Distance เป็นตัวตั้งอัลกอริทึมนี้
จะช่วยวิเคราะห์ความหนาแน่นได้หลายมิติ มากกว่าที่ DBSCAN ทำได้

