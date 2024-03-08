# Prediction of myocardial infarction using a combined generative adversarial network model and feature-enhanced loss function

<div>
    Shixiang Yu, Siyu Han, Mengya Shi, Makoto Harada, Jianhong Ge, Xuening Li, Xiang Cai, Margit Heier, Gabi Karstenmüller, Karsten Suhre, Christian Gieger, Wolfgang Koenig, Wolfgang Rathmann, Annette Peters, Rui Wang-Sattler*
</div>


# :corn: Code

## 1_Normalization

Normalize the dataset, and save the processed dataset.


## 2_balance_C

Load the MI (Myocardial Infarction) and Non_MI (Non-Myocardial Infarction) datasets and their labels, which are stored as .npy files.

- K-means clustering:
  Use the K-means algorithm to cluster the MI and Non_MI datasets into three clusters each.

- Data split:
  Split each cluster group independently into training, validation, and test sets with a ratio of training:validation:test = 64%:16%:20% (first split 20% of the entire data as the test set, and then split 20% of the remaining 80% data as the validation set).

- Combine datasets:
  Recombine the data from different cluster groups to form the overall training, validation, and test sets. Perform this step for both MI and Non_MI data.

- Save datasets:
  Save the finally formed training, validation, and test sets and their corresponding labels as .npy files for future use.
  The code is mainly used for data preprocessing and preparation. Through clustering and splitting methods, it prepares for the subsequent machine learning or deep learning model training.



## 3_GAN_sick-AE

Data Augmentation and Modeling of Myocardial Infarction (MI) Data using Generative Adversarial Networks (GANs)

- Data Loading and Preprocessing:
  Load the preprocessed MI and Non-MI data.
  Merge the MI training and validation data to increase the sample size for training.
- Model Definition:
  Generator (G): A generator with an autoencoder structure is defined to generate new data samples.
  Discriminator (D): A fully connected neural network is defined as the discriminator to distinguish between generated and real data.
- Model Training:
  Train the generator and discriminator alternately in each step.
  Use BCELoss to calculate the loss, and Adam optimizer is selected.
  Specifically, the training of the generator is divided into two steps: the first is to optimize the similarity between the generated images and the input through reconstruction loss, and the second is to further optimize the quality of the generated images through the feedback of the discriminator.
- Data Generation:
  During and after training, use the trained generator to generate new MI data samples and save them as .npy files.
- Model Saving and Loading:
  After training, save the generator and discriminator models, which can be loaded later for performance evaluation.
- Performance Evaluation:
  Evaluate the performance of the GAN by the classification accuracy of the discriminator on Non-MI and MI data, and save the results.

## 4_AE_S4

Data Loading and Preparation: Loads data from "S4AfterNor.csv" and splits it into training and testing datasets. It also loads additional datasets (MI and Non_MI) saved as numpy files.

- Parameter Setup: Defines learning rate, epochs, batch size, and other parameters for training the AE model.

- Model Design: Defines an AutoEncoder with an encoder and a decoder. The encoder compresses the data to a lower-dimensional space (in this case, 3 dimensions), and the decoder attempts to reconstruct the data from this compressed representation.

- Training the AE: Trains the AutoEncoder using the mean squared error loss and Adam optimizer. It iterates through epochs, feeding batches of data to the model, and optimizes the model to minimize the reconstruction loss.

- Data for Visualization: Loads generated data for MI and combines it with original MI data to prepare for visualization.

- Visualization: Visualizes the low-dimensional representation (encoded data) obtained from the AE in a 3D plot. It uses colors to differentiate between data points based on their labels.

- Saving Model and Outputs: Saves the trained AutoEncoder model to a file named "autoencoder.pkl" and saves a visualization of the encoded data as "encoded_S4.tif".


## 5_prediction

Utilize SMOTEENN for data augmentation and train a Deep Neural Network (DNN) for classification purposes.

- Data Loading: The code begins by loading multiple datasets related to MI and Non_MI from numpy files. 

- Data Augmentation: It employs the SMOTEENN, a combination of Synthetic Minority Over-sampling Technique (SMOTE) and Edited Nearest Neighbors (ENN), to augment the training data. This is done to address the class imbalance issue by generating synthetic samples of the minority class (MI) and cleaning overlapping samples between classes.

- Model Training Preparation: Setting up the data loaders with specified batch sizes, and defining hyperparameters for the training process.

- Neural Network Design: DNN is defined with fully connected layers. The model is aimed at classifying MI and Non_MI cases based on the input features.

- Training and Validation: The DNN model is trained over several epochs with the augmented and original data. It evaluates the model's performance on the validation set after each epoch to monitor improvement and potentially save the best-performing model.

- Model Evaluation: After training, the model's performance is evaluated on a separate test dataset. Repeated calculation accuracy multiple times (e.g. 10 times) to ensure the reliability of the results.

  



## 6_Sensitivity

Evaluates the sensitivity of variables datasets using DNN.

- Data Loading: Loads datasets along with their labels from numpy files.

- Average Calculation: Computes the average value for each variable (feature) across all samples in        datasets.

- Model Reloading: Defines DNN structure and reloads a pretrained model for sensitivity analysis.

- Sensitivity Analysis: Replaces each variable in the Non-MI dataset with its average value from the MI dataset and vice versa, then feeds the modified data into the DNN model.
  Calculates the change in model output due to this replacement, which is used to estimate the sensitivity of each variable.

- Aggregation and Visualization:
  Repeats the sensitivity analysis process for multiple instances (5 times) and aggregates the results. Saves the aggregated sensitivity scores to a numpy file. Identifies and highlights the top 20 most sensitive variables based on their scores.

- Plotting:
  Plots a scatter graph showing the sensitivity of each variable. Highlights the top 20 most sensitive variables in red. Saves the plot as a TIFF file for further analysis.
  
  

# :raised_hands: Acknowledgements

We express our appreciation to all KORA study participants for their blood donation and time. We thank all participants for their long-term commitment to the KORA study, the staff for data collection and research data management and the members of the KORA Study Group (https://www.helmholtz-munich.de/en/epi/cohort/kora) who are responsible for the design and conduct of the KORA study.

We would like to acknowledge the contribution of OpenAI's ChatGPT language model in providing information and assistance during the preparation of this manuscript. In addition, we thank the professional author services of Springer Nature for proofreading.  

# :rocket: Funding

This project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking (JU) under grant agreement No 821508 (CARDIATEAM). The JU receives support from the European Union's Horizon 2020 research and innovation programme and the European Federation of Pharmaceutical Industries and Associations (EFPIA). 

The German Diabetes Center is supported by the German Federal Ministry of Health (Berlin, Germany) and the Ministry of Science and Culture in North-Rhine Westphalia (Düsseldorf, Germany). This study was supported in part by a grant from the German Federal Ministry of Education and Research to the German Center for Diabetes Research (DZD).

The KORA study was initiated and financed by the Helmholtz Zentrum München – German Research Center for Environmental Health, which is funded by the German Federal Ministry of Education and Research (BMBF) and by the State of Bavaria. Data collection in the KORA study is done in cooperation with the University Hospital of Augsburg.



#  :apple:Data availability

The KORA data are governed by the General Data Protection Regulation (GDPR) and national data protection laws, with additional restrictions imposed by the Ethics Committee of the Bavarian Chamber of Physicians to ensure data privacy of the study participants. Therefore, the data cannot be made freely available in a public repository. However, researchers with a legitimate interest in accessing the data may submit a request through an individual project agreement with KORA via the online portal (https://www.helmholtz-munich.de/en/epi/cohort/kora).
