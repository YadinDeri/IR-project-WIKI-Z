# Wikipedia Search Engine
![image](https://user-images.githubusercontent.com/76015915/212551748-daeb7d77-30ef-4af4-9ddf-a504e8c74468.png)

 

# Description
The aim of this project is about measuring the effectiveness of standard Information Retrieval systems. 
The standard approach to information retrieval system evaluation revolves around the notion of relevant and non-relevant documents.

 

Basic measures for information retrieval effectiveness that are used for document retrieval are Precision and Recall. So using information retrieval systems we define specific queries and then answer these specific queries.

Measure precision and recall values with the standard information retrieval systems. For these experiments we use Wikipedia corpus that’s hold over 6 million of wiki pages.

 

![image](https://user-images.githubusercontent.com/76015915/212551716-f7bb5c11-946d-4f16-a972-7c4be0b7dbe9.png)

 

# 🚩How to start
import request
search = request.get(url="http://34.72.166.196/search",params={"query":"hello world"})

search_title = request.get(url="http://34.72.166.196/search_title",params={"query":"hello world"})

search_body = request.get(url="http://34.72.166.196/search_body",params={"query":"hello world"})

search_anchor = request.get(url="http://34.72.166.196/search_anchor",params={"query":"hello world"})




# 📚 Dadaset
- Entire Wikipedia dump in a shared Google Storage bucket.
- Pageviews for articles.
- Queries and a ranked list of up to 100 relevant results for them.

 


# 📶 Ranking methods
- Cosine Similarity using TF-IDF - on the body of articles.
- Binary ranking using VSM - on the title and anchor text of articles.
- BM25 - calculating the score of each part in the articles and them merge the results.
- Page Rank
- Page Views

 

# 💡 Platforms
- PyCharm - Python 
- Google Colaboratory
- Google Cloud Platform
- VM in Compute Engine

 


# 📡 Engine's domain
http://35.226.44.201:8080

 

Google storage, a link to our project bucket:  https://console.cloud.google.com/storage/browser/ir-project-z

 

# 📎 Create by:

 

▶️ Yadin Deri : YadinDe@post.bgu.ac.il

▶️ Eden Tzarfaty : edenrivk@post.bgu.ac.il

 

 

   ![image](https://user-images.githubusercontent.com/76015915/212553127-10007d05-f839-42d7-9fc7-c35bf8f742a7.png)
