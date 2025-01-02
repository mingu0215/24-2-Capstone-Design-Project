## Real-time Mental Health Detection via Chatbot
#### We developed an application for diagnosing mental illnesses based on user input. By extracting data through natural language processing, we trained a model using contextual information derived from text, such as symptoms, to develop the CURE model.
---
### [Reference] 
##### __"CURE: Context and Uncertainty-Aware Mental Disorder Detection, EMNLP 2024"__
##### https://github.com/gyeong707/EMNLP-2024-CURE
---
This is overview of our project
![This is flow of our project](https://github.com/mingu0215/24-2-Capstone-Design-Project/blob/main/Workflow.png)

***

### Data
1. Ministry of Health and Welfare National Mental Health Information Portal
  - Extracting symptom keyword combinations
  - Calcuate weights by combining word frequency and TF-IDF using n-gram, with the top 3% weight as threshold(정신질환 증상 추출 및 유형 분류.ipynb)

2. Naver Knowledge-iN posts
  - Collecting [post questions&physicians answers] in the mental health category
  - Filtering only sentences containing keyword combinations in top 3% symptom dictionary to train the model
(네이버 지식인 증상 문장 선별.ipynb)
![Filtering sentences](https://github.com/mingu0215/24-2-Capstone-Design-Project/issues/3#issue-2765468504)

For efficient filtering, annotation tasks were performed.
1. Annotation1
  - [T / F / Uncertain] Annotation Based on the question of posts
   
![Annotation1](https://github.com/mingu0215/24-2-Capstone-Design-Project/issues/4)

2. Annotation2
  - Mental Disease Name Annotation
  - Symptoms Annotation

![Annotation2](https://github.com/mingu0215/24-2-Capstone-Design-Project/issues/5)

Define kinds of mental disease to detect
  - 16 diseases, including "Non-disease"
  - 48 symptoms
![Final Disease](https://github.com/mingu0215/24-2-Capstone-Design-Project/blob/main/Disease.png)

***

### Feature Extraction
1. Symptom Identification
  - Using GPT prompt engineering(gpt-4o)
  - Predict the sympotoms in the post text
  - Calculate "mean_uncertainty" for fusion using "SGNP"

2. Context Extraction
  - Using GPT prompt engineering(gpt-4o)
  - Extraxt 'cause', 'frequency', 'duration', 'age', 'affects'(social, occupational, academic, life-threatening) information of users'
  - The form of context information: categorial(0, 1, 2)&evidence(text)

*** 

### Uncertainty-Aware Decision Fusion(CURE)
![model](https://github.com/mingu0215/24-2-Capstone-Design-Project/blob/main/model.png)
  - Using model 'BERT', 'SYMP'
  - With calculated uncertainty(SNGP), model fusion into CURE

[Post]
- text_input_ids (post text input)
- text_attention_mask

[Context]
- factor_input_ids (context evidence)
- factor_attention_mask
- factors (context categorial tensor)

[Symptom]
- symptom
- symptom_uncertainty

### [MDD Model]
**BERT post -> input: post text**

**BERT context -> input: context evidence + symptoms**

**SYMP sympotm -> input: symptoms vector**

**SYMP context -> input: context category + symptoms**

### Evaluation
![Evaluation](https://github.com/mingu0215/24-2-Capstone-Design-Project/blob/main/Evaluation.png)
