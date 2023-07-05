import pandas
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# reading input data
body_measurements_df = pandas.read_csv('Body Measurements _ original_CSV.csv')

with open('user_input_data.txt', 'r') as file:
    try:
        user_data = [[float(s.strip().split(':')[1]) for s in file.readlines()[2:]]]
    except:
        print('Invalid input file')
        exit()


# separating inputs
input_fields = ['Gender', 'HeadCircumference', 'ShoulderWidth', 'ChestWidth', 'Belly', 'Waist', 'Hips',
                'ArmLength',
                'ShoulderToWaist', 'WaistToKnee', 'LegLength', 'TotalHeight']
inputs = np.array(body_measurements_df[input_fields])

# separating the target
target = np.array(body_measurements_df['Age'])

# creating linear regression model
model = LinearRegression()

# fitting inputs and target in model
model.fit(np.nan_to_num(inputs), np.nan_to_num(target))

# predicting age from user input data
predicted_age = int(model.predict(user_data))
rmse = np.sqrt(np.mean(np.square(target - predicted_age)))
print(predicted_age, '+-', rmse)

# building linear regression equation
regr_equation = 'Age = '
coefs = model.coef_
for i in range(len(input_fields)):
    regr_equation += f'({coefs[i]})*({input_fields[i]}) + '

regr_equation += str(model.intercept_)

print(regr_equation)

# plotting
for i in input_fields:
    tmp_model = LinearRegression()
    tmp_model.fit(np.nan_to_num(body_measurements_df[[i]]), np.nan_to_num(body_measurements_df['Age']))
    func = lambda x: tmp_model.coef_ * x + tmp_model.intercept_
    sns.set_style('darkgrid')
    plt.title(f'Age based on {i} (unit: inch)')
    plt.scatter(body_measurements_df[i], body_measurements_df.Age)
    plt.plot(body_measurements_df[i], func(np.nan_to_num(body_measurements_df[[i]])), color='r')
    plt.xlabel(i)
    plt.ylabel('Age')
    plt.legend(['real', 'predicted'])
    plt.show()


"""
// Seyed Mohsen Razavi Zadegan
// 40030489
// Github link (private until deadline :) ): https://github.com/MohsenRazavi/numeric-analysis-project 
"""