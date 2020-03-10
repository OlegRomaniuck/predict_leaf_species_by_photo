# Predict species by photo of leaf
In this project I'll try using classical methods from ML predict a plant species by photo of leaf.  
Model trained on photo getting from: [**data set**](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)
and support such type of species:  
*Tomato  
*Grape  
*Apple  
*Pepper  
*Strawberry  
*Squash  
*Corn  
*Peach  
*Soybean  
*Potato  
*Raspberry  
*Cherry  
*Orange  
*Blueberry  
:herb:  :fallen_leaf:  :leaves:  
 
##**Features**  
Pick out features:  
* gradient features: laplacian_var, sobelx_var, sobely_var, sobelx8u_var, sobelx64f_var
* geometrical: area, perimeter, physiological_length, physiological_width, aspect_ratio, rectangularity, circularity
* color: mean_r, mean_g, mean_b, stddev_r, stddev_g, stddev_b, contrast, correlation, entropy

Models was build based on:
* RandomForestClassifier algorithm  
* XGBoost algorithm  

##**Comparative table**  

|    | RandomForestClassifier  | XGBoost         |
-----| ------------------------ | ------------- |
| Best Parameters    | max_depth=60, max_features=14, min_samples_leaf=2, min_samples_split=7, n_estimators=100 | learning_rate=0.5, max_depth=5, num_class=14, subsample=1, colsample_bytree=1, n_estimators=200, eta=1 |
| Criterion  | "gini" | ["merror", "mlogloss", "auc"]  |  
| Best score   | 0.8430940111538154  | 0.8873304479467492  |  
| Accuracy  | 0.8536967130098517 | 0.8744130374735292  |  
| CV accuracy score  | 79.38%  | 83.04% |  
| F1 Micro Average   | 0.8536967130098517  | 0.8744130374735292  |  
| Confusion matrix  | conf_tree.png  | conf_xgboost.png  |  

##Result: Telegram bot
Name: **@cegorach_executor_bot**   
Api:
/start some help for user
/models – list of using models
/accept_model pass what type of model do you want to use, default is “VGG”
Simple pass model name without command
Simple pass picture and wait result
Response is: name of tree or vegetable
 




