# Comparing Performance of Support-Vector Machine and Logistic Regression

## Description of Task

This notebook is a solution to the following task:

Compare the performance of a Support-Vector Machine with that of Logistic Regression.
Try to optimize both algorithms' parameters and determine which one is best for thisdataset. 
At the end of the analysis, you should have chosen an algorithm and its optimal set of parameters:
write this choice explicitly in the conclusions of your notebook.

## Description of Dataset

This dataset is composed of 1300 samples with 25 features each. 
The first column is the sample id. The second column in the dataset represents the label. 
There are 2 possible values for the labels. The remaining columns are numeric features.


### Importing Dataset:


    #Observations  #Features    #NA Values  #Duplicates    label 
    ----------------------------------------------------------------
             1300         25          None         None    49.7 %


This quickly verifies that we have 1300 observations and 25 features. In addition, we see that there are no NA value or duplicated rows and that there is a ~50:50 proportion between the two labels in the dataset. Next, let's perform some dataset analysis.

### Exploratory Data Analysis

In this section we will perform brief exploration into our data, looking at summary statistics, plotting the histogram spread of each feature, and looking at the correlation between each feature. This will help us to understand what the nature of the data and give us the opportunity to notice relationships in the data.




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
      <th>feature_3</th>
      <th>feature_4</th>
      <th>feature_5</th>
      <th>feature_6</th>
      <th>feature_7</th>
      <th>feature_8</th>
      <th>feature_9</th>
      <th>feature_10</th>
      <th>...</th>
      <th>feature_16</th>
      <th>feature_17</th>
      <th>feature_18</th>
      <th>feature_19</th>
      <th>feature_20</th>
      <th>feature_21</th>
      <th>feature_22</th>
      <th>feature_23</th>
      <th>feature_24</th>
      <th>feature_25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>...</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
      <td>1300.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.338439</td>
      <td>-0.151954</td>
      <td>-0.185151</td>
      <td>0.311299</td>
      <td>0.279897</td>
      <td>0.003881</td>
      <td>0.178295</td>
      <td>-0.542319</td>
      <td>0.002054</td>
      <td>-0.004932</td>
      <td>...</td>
      <td>-0.429254</td>
      <td>-0.347421</td>
      <td>-0.502387</td>
      <td>0.433979</td>
      <td>0.204415</td>
      <td>-0.582756</td>
      <td>-1.053712</td>
      <td>0.107390</td>
      <td>-0.032372</td>
      <td>-0.374981</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.427770</td>
      <td>1.670306</td>
      <td>0.981991</td>
      <td>2.404899</td>
      <td>1.223137</td>
      <td>1.136962</td>
      <td>1.051852</td>
      <td>1.701212</td>
      <td>1.200918</td>
      <td>1.043793</td>
      <td>...</td>
      <td>1.861464</td>
      <td>1.435908</td>
      <td>1.197181</td>
      <td>2.039115</td>
      <td>1.199795</td>
      <td>0.666962</td>
      <td>2.567385</td>
      <td>1.100071</td>
      <td>1.963108</td>
      <td>1.321986</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-5.577567</td>
      <td>-6.132911</td>
      <td>-3.953686</td>
      <td>-8.213834</td>
      <td>-3.473453</td>
      <td>-4.393443</td>
      <td>-3.724541</td>
      <td>-5.207074</td>
      <td>-4.245091</td>
      <td>-3.596323</td>
      <td>...</td>
      <td>-8.235007</td>
      <td>-4.290421</td>
      <td>-5.107172</td>
      <td>-5.532940</td>
      <td>-4.604010</td>
      <td>-2.660711</td>
      <td>-11.418329</td>
      <td>-3.258889</td>
      <td>-6.199655</td>
      <td>-4.451162</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.211653</td>
      <td>-1.213953</td>
      <td>-0.827902</td>
      <td>-1.101432</td>
      <td>-0.557659</td>
      <td>-0.720631</td>
      <td>-0.580719</td>
      <td>-1.774042</td>
      <td>-0.809177</td>
      <td>-0.701703</td>
      <td>...</td>
      <td>-1.537929</td>
      <td>-1.293797</td>
      <td>-1.256219</td>
      <td>-1.062781</td>
      <td>-0.572558</td>
      <td>-1.034037</td>
      <td>-2.328044</td>
      <td>-0.625966</td>
      <td>-1.419217</td>
      <td>-1.263837</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.248385</td>
      <td>-0.031338</td>
      <td>-0.155223</td>
      <td>0.535549</td>
      <td>0.231154</td>
      <td>-0.039458</td>
      <td>0.139593</td>
      <td>-0.575434</td>
      <td>0.039064</td>
      <td>-0.010828</td>
      <td>...</td>
      <td>-0.384438</td>
      <td>-0.331566</td>
      <td>-0.463897</td>
      <td>0.551644</td>
      <td>0.237767</td>
      <td>-0.570126</td>
      <td>-0.624524</td>
      <td>0.079981</td>
      <td>-0.016846</td>
      <td>-0.378328</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.640682</td>
      <td>0.916374</td>
      <td>0.530636</td>
      <td>2.052738</td>
      <td>1.067619</td>
      <td>0.772724</td>
      <td>0.886920</td>
      <td>0.562960</td>
      <td>0.798489</td>
      <td>0.686624</td>
      <td>...</td>
      <td>0.775868</td>
      <td>0.562251</td>
      <td>0.306079</td>
      <td>1.949728</td>
      <td>1.016510</td>
      <td>-0.132454</td>
      <td>0.632363</td>
      <td>0.838287</td>
      <td>1.310980</td>
      <td>0.439124</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.720374</td>
      <td>5.298651</td>
      <td>2.773804</td>
      <td>6.133258</td>
      <td>4.615081</td>
      <td>4.220863</td>
      <td>3.983645</td>
      <td>5.448051</td>
      <td>3.626921</td>
      <td>3.492904</td>
      <td>...</td>
      <td>5.370952</td>
      <td>4.217335</td>
      <td>3.630840</td>
      <td>5.730333</td>
      <td>4.078821</td>
      <td>1.407933</td>
      <td>5.377527</td>
      <td>3.244948</td>
      <td>8.961113</td>
      <td>6.407959</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>




    
![png](README_files/README_8_0.png)
    


From the histograms and summary statistics we can see that we have symmetrical data. In addition, we can see that our data is normally distributed with mean near 0 and a standard deviation of most 2.6. Next, the correlation heatmap will tell us how our features relate to one another.


    
![png](README_files/README_10_0.png)
    


### Unsupervised Learning

Next, we can perform Principal Component Analysis to look at how much we can reduce dimensions and retain the information of our data. We will look at how many components are needed to retain the data's variance, specifically looking at 80, 90, and 95 % variance. We will also then plot the variance explained by each component to see a visual representation of this.

       Variance Explained       #Components   Dimensionality Kept 
    ---------------------------------------------------------------
               80%                   7               28.00%       
               90%                  13               52.00%       
               95%                  16               64.00%       



    
![png](README_files/README_13_0.png)
    


Now that we have reduced the data, we can take our first three principal components and plot the points in the space spanned by these three components. We are also going to color the points according to their label to see if there is a clear split between the two labels of the data.

    
![png](README_files/README_15_1.png)
    


### Supervised Learning

Now that we have some idea of the nature of our data, we can move on to our models.

Let us prepare and scale the data and define our models. We will be using logistic regression, support vector machine with a linear kernel, and then we will extend our analysis to SVM with other kernels too.

### Logistic Regression

In our parameter grid we will experiment with different values for C, varying the regularization of the classifier, as well as the l1_ratio, which weighs our l1 vs l2 norms regularization.

        param_C  param_l1_ratio  mean_test_score  rank_test_score
    0      0.01             0.5         0.811538               10
    1      0.01             0.7         0.791346               11
    2      0.01             0.9         0.782692               12
    3      0.10             0.5         0.823077                1
    4      0.10             0.7         0.822115                2
    5      0.10             0.9         0.817308                8
    6      1.00             0.5         0.821154                6
    7      1.00             0.7         0.818269                7
    8      1.00             0.9         0.817308                8
    9     10.00             0.5         0.822115                2
    10    10.00             0.7         0.822115                2
    11    10.00             0.9         0.822115                2
    Best parameters: {'C': 0.1, 'l1_ratio': 0.5}
    Best score: 0.823076923076923


The logistic regression gives us a score of 0.82, which is a good start, but let's try to make use of polynomial features to see if this will give us a clearer split in our data.
Logistic regression (and LSVC) works best with linearly separable data. Through polynomial features, we perform a nonlinear transformation of the data which might facilitate finding this linear separation. 

Indeed, we notice an improved performance of 0.94. Note that the initial values for the C parameter were [0.01, 0.1, 1, 10], yet upon seeing that 0.1 performed the best, I decided to look around 0.1, changing to [0.1, 0.3, 0.5].


### Support Vector Machine (linear kernel)

Next, let's move to the linear kernel Support Vector Machine, where again we experiment with the C parameter, and this time with the l1 and l2 penalties separately.

       param_C param_penalty  mean_test_score  rank_test_score
    0     0.01            l1         0.811538               10
    1     0.01            l2         0.818269                9
    2     0.10            l1         0.819231                8
    3     0.10            l2         0.824038                1
    4     1.00            l1         0.822115                7
    5     1.00            l2         0.824038                1
    6    10.00            l1         0.824038                1
    7    10.00            l2         0.824038                1
    8   100.00            l1         0.824038                1
    9   100.00            l2         0.824038                1
    Best parameters: {'C': 0.1, 'penalty': 'l2'}
    Best score: 0.8240384615384615


As expected, given both classifiers are geared towards linearly separable data, we see a similar performance for the LinearSVC on the basic data.As we did for Logistic Regression, let's try performing LinearSVC on polynomial features as well.

       param_C param_penalty  mean_test_score  rank_test_score
    0     0.01            l1         0.909615                3
    1     0.01            l2         0.932692                2
    2     0.10            l1         0.944231                1
    3     0.10            l2         0.896154                4
    4     1.00            l1         0.888462                5
    5     1.00            l2         0.874038                6
    6    10.00            l1         0.867308                7
    7    10.00            l2         0.865385                9
    8   100.00            l1         0.866346                8
    9   100.00            l2         0.864423               10
    Best parameters: {'C': 0.1, 'penalty': 'l1'}
    Best score: 0.9442307692307692


Again, a better performance on the polynomial features. Given that so far we have only looked at linear kernel, let's expand our exploration to the Gaussian and sigmoid kernels.

### Support Vector Machine

In our parameter grid we will define the regularization parameter and the kernel, we will use the default values for gamma.

        param_C param_kernel  mean_test_score  rank_test_score
    0      0.01          rbf         0.707692               14
    1      0.01      sigmoid         0.799038               10
    2      0.01         poly         0.630769               15
    3      0.10          rbf         0.902885                4
    4      0.10      sigmoid         0.802885                9
    5      0.10         poly         0.821154                8
    6      1.00          rbf         0.934615                1
    7      1.00      sigmoid         0.740385               11
    8      1.00         poly         0.874038                5
    9     10.00          rbf         0.929808                2
    10    10.00      sigmoid         0.712500               13
    11    10.00         poly         0.850962                6
    12   100.00          rbf         0.914423                3
    13   100.00      sigmoid         0.720192               12
    14   100.00         poly         0.828846                7
    Best parameters: {'C': 1, 'kernel': 'rbf'}
    Best score: 0.9346153846153846


Not quite as good as the linear kernel, put a good performance nonetheless. Next, let's take the better performing kernel, rbf, and let's try a few different gamma values:

        param_C param_kernel  param_gamma  mean_test_score  rank_test_score
    0      0.01          rbf         0.01         0.635577                9
    1      0.01          rbf         0.10         0.581731               10
    2      0.01          rbf         1.00         0.499038               14
    3      0.10          rbf         0.01         0.843269                8
    4      0.10          rbf         0.10         0.891346                7
    5      0.10          rbf         1.00         0.499038               14
    6      1.00          rbf         0.01         0.898077                6
    7      1.00          rbf         0.10         0.928846                2
    8      1.00          rbf         1.00         0.562500               11
    9     10.00          rbf         0.01         0.931731                1
    10    10.00          rbf         0.10         0.924038                3
    11    10.00          rbf         1.00         0.562500               11
    12   100.00          rbf         0.01         0.922115                5
    13   100.00          rbf         0.10         0.924038                3
    14   100.00          rbf         1.00         0.562500               11
    Best parameters: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
    Best score: 0.9317307692307694


## Performance

Having performed GridSearch to find the best parameters for different variations of our models, let's plot all of our models' ROC curve:


    
![png](README_files/README_31_0.png)
    


Finally let's rank all of our models:

    Position	| Classifier					| Accuracy
    --------------------------------------------------------------------------------
    1		| Support Vector Machine        		| 0.95   
    2		| SVM w/ RBF Kernel Detailed    		| 0.95   
    3		| LinearSVC Polynomial Features 		| 0.94   
    4		| Logistic Reg Polynomial Features		| 0.93   
    5		| Logistic Regression           		| 0.82   
    6		| LinearSVC                     		| 0.82   


## Test Set

Given that my task was to look at Logistic Regression and Linear SVC, I will be implementing the best model out of that category. As seen above, the best performing model was the LinearSVC with polynomial features. This means that before using the model to predict the labels, we need to first create the polynomial features.

## Conclusion

The task was to compare the performance of Logistic Regression and LinearSVC. I did this through a GridSearchCV in which I tried different parameters. First I split the data into a test and a train portion, and performed cross-validation through the training data, with 5 partitions.The initial choice of parameters gave me a basis to then explore with more care. In the Logistic Regression, I experimented with different **C  and l1ratio values**. Starting with a range of **different magnitudes for C**, I was able to pinpoint the range which translated into the best performance. I then tested different parameters closer to this range, which provided marginal returns but still improved the accuracy. I also noticed that it was useful to run the data through polynomial features, because it was **providing a nonlinear transformation**, increasing the range of my classifiers. For LinearSVC, I experimented with **different C parameters and penalties**. I also used polynomial features for the LinearSVC. For the last model, since LinearSVC only uses the linear kernel, I tried the Support Vector Machine with the **radial, sigmoid, and polynomial kernel**. The rbf kernel performed the best, therefore a final extension was done, looking at **different values for gamma**. 

Finally, in order to visualize the accuracy, the **ROC curve** was plotted for each best model, which tells us how well binary classification models perform. By plotting sensitivity by (1 - specificity) across different thresholds, we get a visual showcase of the comparitive perfomance of the models. The area under the curve, **AUC**, gives us the overall performance of the model, which is also an indicator that can be compared.

