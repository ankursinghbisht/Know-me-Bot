# My Digital Clone

A Large Language Model (LLM), fine-tuned over my knowledge base.
<img src="https://github.com/ankursinghbisht/My-Digital-Clone/assets/112644477/391c341f-d9ee-4835-bd1d-d637254ce1a1" alt="image" width="600"/>

## Introduction

A chatbot constructed from a large language model that can imitate my conversational style through using my personal knowledge. It works on a type of architecture called "transformer."


## Architecture of Transfomers
<img src="https://github.com/ankursinghbisht/My-Digital-Clone/assets/112644477/00453b07-0f98-4ef5-addc-986fbafe5619" alt="image" width="300"/>

## About Transfomers


The Transformer architecture is a type of neural network architecture that has revolutionized natural language processing (NLP) tasks. It's made up of several key components:

### Input Embeddings: 
At the start, words or tokens in a sentence are converted into numerical vectors (embeddings) that capture their meanings and relationships.

### Encoder:
This part processes the input embeddings. It consists of multiple layers, typically called encoder layers. Each encoder layer has two main components: the multi-head self-attention mechanism and a feedforward neural network. The self-attention mechanism helps the model focus on different parts of the input sequence when encoding it.

### Decoder: 
This part generates the output sequence. Like the encoder, it also has multiple layers, called decoder layers. Each decoder layer has three main components: masked multi-head self-attention, encoder-decoder attention, and a feedforward neural network. The decoder uses the encoded representation from the encoder along with its own self-attention mechanism to generate the output sequence.

### Multi-Head Attention: 
This is a mechanism that allows the model to focus on different parts of the input sequence simultaneously. It calculates attention scores between each pair of words in the sequence, providing a weighted representation of the entire sequence for each word.

### Feedforward Neural Network: 
This is a standard neural network layer that applies a series of linear transformations followed by non-linear activation functions to transform the representations of the words.

### Positional Encoding: 
Since Transformers don't have inherent notions of word order, positional encodings are added to the input embeddings to provide information about the positions of words in the sequence.

Overall, the Transformer architecture excels at capturing long-range dependencies in sequences, making it particularly effective for tasks like machine translation, text summarization, and language understanding. It's highly parallelizable, which makes it easier to train on large datasets using modern hardware like GPUs and TPUs.




## Key features

- _Question-answering_: Implementing a robust question-answering system allows users to ask queries in natural language, and the model retrieves relevant answers from its knowledge base.

- _Context memory_: The context memory feature enables the model to retain information from previous interactions, improving its ability to maintain coherent and contextually relevant conversations over extended dialogues.

- _Transformer Architecture_: Leveraging the Transformer architecture, specifically through frameworks like PyTorch and the Transformers library, facilitates the development of highly performant and scalable language models capable of handling complex language tasks.

- _Custom Dataset Training_: Training the language model on a custom dataset tailored to specific domains or user preferences enhances its proficiency in understanding and generating contextually appropriate responses for a wide range of topics and conversations.





## Chat Screenshots
<img src="https://github.com/ankursinghbisht/My-Digital-Clone/assets/112644477/2ec47c71-3193-45cf-a87e-8d8af89bf6f8" alt="image" width="500"/>
<img src="https://github.com/ankursinghbisht/My-Digital-Clone/assets/112644477/ff3c317f-3808-4ff8-8c1c-9c5ba6193850" alt="image" width="500"/>
<img src="https://github.com/ankursinghbisht/My-Digital-Clone/assets/112644477/b678364f-3a77-4904-8a18-2fe5006d4c22" alt="image" width="500"/>
<img src="https://github.com/ankursinghbisht/My-Digital-Clone/assets/112644477/644fa32a-463a-488e-b6ff-9e3bf1b931c2" alt="image" width="500"/>
<img src="https://github.com/ankursinghbisht/My-Digital-Clone/assets/112644477/e71cb172-ec4e-4d97-8403-05b5eb330b2c" alt="image" width="500"/>


## Why I Built it?
I was inspired to create my digital clone project by my excitement with the latest developments in NLP. I wanted to build a customised digital assistant that could emulate my speech patterns and have meaningful conversations, motivated by the promise of big language models. By using my knowledge base to improve a language model and implementing it in an approachable online application, I aimed to push the limits of artificial intelligence.



## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/ankursinghbisht?tab=repositories)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ankursinghbisht/)

