# Employee Attrition Prediction

This project aims to predict employee attrition using machine learning techniques. It utilizes a dataset containing various employee attributes and applies data preprocessing, model training, evaluation, and interpretation using SHAP values to understand the factors influencing employee turnover.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Interpreting Results](#interpreting-results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Data loading and preprocessing
- Model training using Random Forest
- Model evaluation including accuracy, confusion matrix, and classification report
- Feature importance visualization
- SHAP values for model interpretability
- Correlation matrix for initial data understanding

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP

## Dataset

The dataset used in this project contains the following features:

- `EmployeeID`: Unique identifier for each employee
- `Age`: Age of the employee
- `MonthlyIncome`: Monthly income of the employee
- `YearsAtCompany`: Number of years the employee has worked at the company
- `JobSatisfaction`: Satisfaction rating of the employee (scale from 1 to 5)
- `Attrition`: Target variable indicating whether the employee has left the company (`Yes` or `No`)

An example of the dataset is as follows:

```csv
EmployeeID,Age,MonthlyIncome,YearsAtCompany,JobSatisfaction,Attrition
1,45,6500,12,4,No
2,34,3200,3,2,Yes
3,29,2700,1,3,Yes
4,41,5200,8,4,No
5,25,3500,2,2,Yes
6,50,8000,15,3,No
7,39,4400,7,4,No
8,31,3900,5,3,Yes
9,28,3000,2,2,Yes
10,49,7000,10,4,No
```

