# A System for Measuring Diabetes Progression (SMDP)

### 1. Summary

Diabetes become much more prevalent with high carbohydrate consumption, which may casue many derived diseases. Thus, in many medical institude, observation based on the diabete patients biomedical indexes are carried out, and leave precious records for analysis and research purpose. One of these records, which also referred as datasets, are from Bradley Efron etc. in Stanford, recorded in 2004. Based on this dataset, a regression algorithm (model) will be implemented to better predict the patients diabetes level, which in a more formal way as progression, with archieved biomediacal indexes. For convenient perspective, we call this model in acronym form as  dia_reg, which shorted for diabetes regressioner.

### 2. Overview of Biomedical Indexed (Input Features)
1. age: age in years
1. sex: gender distinction
1. bmi: body mass index
1. bp: average blood pressure
1. s1:  tc, total serum cholesterol
1. s2: ldl, lowensity lipoproteins
1. s3: hdl, highensity lipoproteins
1. s4: tch, total cholesterol / HDL
1. s5: ltg, possibly log of serum triglycerides level
1. s6: glu, blood sugar level

***Example Input:***

- A vector of biomedical index with the above 10 features.

***Example Output:***

- A scalar value as diabetes regression score.

### 3. Evaluation result

| model | RMSE | R2 |
| ------ | ------ | ------ |
| Multi-linear | 6.228182457896 | 0.5655252353855922 |
| XGBoost | 6.41393974134736 | 0.5064397470314956 |

### 4. Method to run application

```sh
cd SMDP
python main.py
```

