## Real-time Mental Health Detection via Chatbot
#### We developed an application for diagnosing mental illnesses based on user input. By extracting data through natural language processing, we trained a model using contextual information derived from text, such as symptoms, to develop the CURE model.
---
### [Reference] 
##### __"CURE: Context and Uncertainty-Aware Mental Disorder Detection, EMNLP 2024"__
##### https://github.com/gyeong707/EMNLP-2024-CURE
---
This is overview of our project
![This is flow of our project](https://github.com/mingu0215/24-2-Capstone-Design-Project/blob/main/Workflow.png)

### Data
1. Ministry of Health and Welfare National Mental Health Information Portal
  - Extracting symptom keyword combinations
  - Calcuate weights by combining word frequency and TF-IDF using n-gram, with the top 3% weight as threshold(정신질환 증상 추출 및 유형 분류.ipynb)

2. Naver Knowledge-iN posts
  - Collecting [post questions&physicians answers] in the mental health category
  - Filtering only sentences containing keyword combinations in top 3% symptom dictionary to train the model(네이버 지식인 증상 문장 선별.ipynb)




