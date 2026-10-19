| method     |   n_features | features                                                                              |   cv_rmse |   test_rmse |   test_r_squared |
|:-----------|-------------:|:--------------------------------------------------------------------------------------|----------:|------------:|-----------------:|
| Backward   |           10 | zn | chas | nox | rm | dis | rad | tax | ptratio | black | lstat                      |    5.1033 |      4.7920 |           0.7449 |
| Exhaustive |           10 | zn | chas | nox | rm | dis | rad | tax | ptratio | black | lstat                      |    5.1033 |      4.7920 |           0.7449 |
| Forward    |           10 | zn | chas | nox | rm | dis | rad | tax | ptratio | black | lstat                      |    5.1033 |      4.7920 |           0.7449 |
| Lasso      |           13 | crim | zn | indus | chas | nox | rm | age | dis | rad | tax | ptratio | black | lstat |    5.1539 |      4.8411 |           0.7396 |
| RFECV      |           13 | crim | zn | indus | chas | nox | rm | age | dis | rad | tax | ptratio | black | lstat |    5.1539 |      4.8463 |           0.7390 |
