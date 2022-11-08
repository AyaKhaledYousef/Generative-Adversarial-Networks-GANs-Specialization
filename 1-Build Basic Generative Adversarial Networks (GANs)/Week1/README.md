## Machine learning models can be classified into two types of models : ( Discriminative and Generative ) models


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
---------------------------------------------------------------------------------------------------------------------

# Generative Models
## 1- Variational AutoEncoders ( VAE ) :

![VAE](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/VAE.jpg)

- Autoencoders are a type of neural network that learns the data encodings from the dataset in an unsupervised way. 
- It basically contains two parts: the first one is an encoder which is similar to the convolution neural network except for the last layer. 
- The aim of the encoder to learn efficient data encoding from the dataset and pass it into a bottleneck architecture. 
- The other part of the autoencoder is a decoder that uses latent space in the bottleneck layer to regenerate the images similar to the dataset. 
- These results backpropagate from the neural network in the form of the loss function.
- Variational autoencoder is different from autoencoder in a way such that it provides a statistic manner for describing the samples of the dataset in latent space. 
- Therefore, in variational autoencoder, the encoder outputs a probability distribution in the bottleneck layer instead of a single output value.

## 2- Generative Adversarial Network (GAN)

![Generator](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/Generator.png)

- The first step in establishing a GAN is to identify the desired end output and gather an initial training dataset based on those parameters. 
- This data is then randomized and input into the **generator** until it acquires basic accuracy in producing outputs.
- After this, the **generated** images are fed into the **discriminator** along with actual data points from the original concept. 
- The **discriminator** filters through the information and returns a probability between 0 and 1 to represent each image's authenticity (1 correlates with real and 0 correlates with fake). 
- These values are then manually checked for success and repeated until the desired outcome is reached.

![Generator](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/GAN.png)


# Binary Cross Entropy function (BCE)
- Is used for training GANs. It's useful for these models, because it's especially designed for classification tasks, where there are two categories like, real and fake. 

![BCE](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/BCE1.png)

![BCE](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/BCE2.png)

![BCE](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/BCE3.png)

![BCE](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/BCE4.png)

![BCE](https://github.com/AyaKhaledYousef/Generative-Adversarial-Networks-GANs-Specialization/blob/main/1-Build%20Basic%20Generative%20Adversarial%20Networks%20(GANs)/Week1/images/BCE5.png)

## Reference


- [Generative VS Discriminative Models](https://mirror-medium.com/?m=https%3A%2F%2Fmedium.com%2F%40mlengineer%2Fgenerative-and-discriminative-models-af5637a66a3)
- [Deep Understanding of Discriminative and Generative Models in Machine Learning](https://www.analyticsvidhya.com/blog/2021/07/deep-understanding-of-discriminative-and-generative-models-in-machine-learning/#:~:text=Discriminative%20models%20draw%20boundaries%20in,the%20labels%20of%20the%20data.)
- [Variational AutoEncoders](https://www.geeksforgeeks.org/variational-autoencoders/)
- [Generator](https://www.geeksforgeeks.org/generative-adversarial-network-gan/?ref=gcse)
- [GAN](https://www.geeksforgeeks.org/generative-adversarial-network-gan/?ref=gcse)
- [Tensors](https://pytorch.org/docs/stable/tensors.html)
