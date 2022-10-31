
# Generative Models


Machine learning models can be classified into two types of models – **Discriminative** and **Generative** models

## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## 1- Generative :
Generative models are considered as a class of statistical models that can generate new data instances. These models are used in unsupervised machine learning as a means to perform tasks such as.



- Assume some functional form for P(Y), P(X|Y).
- Estimate parameters of P(X|Y), P(Y) directly from training data.
- Use Bayes rule to calculate P(Y |X).
- Examples of (Generative classifiers):

        ‌- Naïve Bayes
        - Bayesian networks
        - Markov random fields
        ‌- Hidden Markov Models (HMM)
        - Generative Adversarial Networks (GANs)



## 2- Discriminative  :

The discriminative model refers to a class of models used in Statistical Classification, mainly used for supervised machine learning. These types of models are also known as conditional models since they learn the boundaries between classes or labels in a dataset.

- Assume some functional form for P(Y|X).
- Estimate parameters of P(Y|X) directly from training data.
- Examples of (Discriminative classifiers):

        ‌‌- Logistic regression
        - Scalar Vector Machine
        ‌- Traditional neural networks
        ‌- Nearest neighbour
        - Conditional Random Fields (CRF)s
    

# Difference between Discriminative and Generative Models:
- Discriminative models draw boundaries in the data space, while generative models try to model how data is placed throughout the space. A generative model focuses on explaining how the data was generated, while a discriminative model focuses on predicting the labels of the data.

- In mathematical terms, a discriminative machine learning trains a model which is done by learning parameters that maximize the conditional probability P(Y|X), while on the other hand, a generative model learns parameters by maximizing the joint probability of P(X, Y).



https://lilianweng.github.io/posts/2018-08-12-vae/
## Reference


- [Generative VS Discriminative Models](https://mirror-medium.com/?m=https%3A%2F%2Fmedium.com%2F%40mlengineer%2Fgenerative-and-discriminative-models-af5637a66a3)
- [Deep Understanding of Discriminative and Generative Models in Machine Learning](https://www.analyticsvidhya.com/blog/2021/07/deep-understanding-of-discriminative-and-generative-models-in-machine-learning/#:~:text=Discriminative%20models%20draw%20boundaries%20in,the%20labels%20of%20the%20data.)

