# uber-lyft-price-prediction-models
Doing price prediction for Uber and Lyft rides by using R

- This project was done as an academic assignment to create machine learning models by using Rstudio. With an interest in the technology of ride-sharing, I chose a dataset from Kaggle to build the models in an effort to predict prices for Uber rides. The [project report](https://github.com/weiaun96/uber-lyft-price-prediction-models/blob/main/Project%20Report.pdf) is also included in the repository.
- The dataset is retrieved from [Kaggle](https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices).

## Results
![](https://github.com/weiaun96/uber-lyft-price-prediction-models/blob/main/Images/Results.JPG)
- The linear regression achieved a good accuracy with R2 = 0.92. However, as a comparison with the other models, RMSE is a better measure to compare between the models. As for the RMSE measure, linear regression achieved RMSE of 2.455. On the other hand, the SVM RBF model with the train-test split of 80:20 ratio achieved the lowest RMSE at 2.087 and managed to reach RMSE of 2.0703 after doing parameter tuning.

- Surprisingly, the ridge and lasso regression did not perform well with only RMSE of 6.32 while the hybrid of them, the elastic regression fare worse with only RMSE of 6.402. This indicates the importance of all the independent variables for the price prediction because the regularized regressions either penalized seemingly unimportant features or make them zero. This method was supposed to improve the model fit but end up it reduces the accuracy of the model.

- As comparison with the other models in literature review, the linear regression in this study performed reasonably good with R2 = 0.92 and RMSE of 2.455. But the regularized regressions did not performed so well with RMSE more than 6 whereby others were in the range of between 3 to 5. SVM model using the RBF kernel seems to be a good model where it achieved the lowest RMSE among all the models used for price prediction.
