## Binary Classification using Neural Network

# About the dataset 
This project utilizes a modified binary classification dataset designed to distinguish between two distinct varieties of rice based on their 
structural features.
  -Task: Binary Classification
  -Classes: Jasmine (1) and Gonen (0)
  -Size: 18,185 total samples | 12 numeric attributes | ~2.03 MB

The model trains on 11 attributes extracted from the rice grains, including:
  -Size metrics: Area, Perimeter, Convex Area, and Equivalent Diameter.
  -Shape/Length metrics: Major/Minor Axis Length, Aspect Ratio, Eccentricity, and Roundness.

# Tech Stack and Implementation Detail 
  -PyTorch : Framework of choice used to design and train the deep learning classification model.
  -Numpy and Pandas : Utilized for handling missing values (dropna), deleting unused variables (e.g., dropping the id column), and splitting variables    into vectors.
  -Scikit-Learn : Implemented to strictly bound measurements between 0 and 1, preventing numeric instability linear calculations.
  -PyTorch Dataset & DataLoader : Used for Cross Validation i.e. to split the data into train, val, and test. Here the size of train is kept at 70%,       test is 15%, and validation is 15% of the entire dataset.

# Model Architecture Overview 
The classification uses a custom Feedforward Deep Neural Network (myModel) with dynamic sizing pipelines designed specifically to consume single tabular numeric feature rows and output exact probability bounds. The network parameters consist of 121 Trainable Parameters in total.

# Operational Flow 
  -Input Vector: Receives an input dimension corresponding to the 10 geometric structural metrics from the normalized dataframe dataset.
  -Hidden Projection Layer: Passes raw vectors through a Fully Connected Linear interface mapping 10 to 10 hidden state features.
  -Logit Extraction Layer: Passes the intermediate representations into a secondary linear layer scaling dimensions down from 10 to 1.
  -Probability Mapping Layer: Applies a non-linear continuous mathematical Sigmoid squashing function to transform raw linear activations directly        into binary metric margins bounded precisely between [0.0, 1.0].

# Optimization & Hyperparameters
  -Loss Metric Function: Binary Cross Entropy Loss (nn.BCELoss()) used to calculate error bounds per batch.
  -Activation Functin : Used Sigmoid function.
  -Optimizer: Adam Optimizer initialized with learning rate setting of 1e-3 (0.001).
  -Training Depth Runtime: Scaled across 10 epochs, calculating batch performance values using raw binary rounding thresholds (prediction.round()).
  -Final Model Accuracy: The model converged rapidly, generating a test verification accuracy evaluation metric of 98.64%.

# Key Findings & Conclusion
  -Rapid Convergence: The model established highly accurate decision boundaries rapidly within the first few epochs, demonstrating that the image-        extracted structural attributes are highly discriminative.
  -Final Performance: The network achieved a final validation and test verification accuracy of 98.64%. 
  -Takeaway: This project successfully demonstrates that deep learning models can achieve near-perfect classification performance on structural/          tabular tabular data when combined with correct feature normalization (`MinMaxScaler`) and optimized gradient descent tools like the Adam optimizer.

  <img width="1553" height="573" alt="image" src="https://github.com/user-attachments/assets/8a6d0b55-b7d4-4da5-8149-5c90566a4e2f" />
