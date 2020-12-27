# Master-Thesis
For full reading of my thesis, you can access from this link (https://essay.utwente.nl/83138/) <br/>

This repository contains my thesis codes. My thesis was related to feature extraction for face recognition system using autoencoder. In this thesis, I tried several autoencoder models such as variational autoencoder, adversarial autoencoder, and wasserstein autoencoder. I also did a little modification on wasserstein autoencoder by changing its network using residual network. This network is called as Resnet-WAE.
The autoencoder results for different latent dimension size are shown in the image below.

![Screenshot](recons_image.png)

As the depth of the network is increased, it can be seen from the results below that the deeper the network is, the higher the accuracy results. This is because the deeper network can extract more features compare to the shallow network.

![Screenshot](ROC_Euclidean.png)
![Screenshot](ROC_FRGC.png)

