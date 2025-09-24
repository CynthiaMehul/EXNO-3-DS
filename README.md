## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, PowerTransformer
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("C:/Users/admin/Documents/AIML/SEM 5/ds/exp3/supermarket.csv")

df.head()

le = LabelEncoder()
df['Branch'] = le.fit_transform(df['Branch'])
df['City'] = le.fit_transform(df['City'])
df['Customer type'] = le.fit_transform(df['Customer type'])
df['Gender'] = le.fit_transform(df['Gender'])

df

ohe = OneHotEncoder(sparse=False)
encoded = ohe.fit_transform(df[['Product line']])
df_ohe = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['Product line']))

df_ohe

payment_order = ['Cash', 'Credit card', 'Ewallet']
enc = OrdinalEncoder(categories=[payment_order])
df['Payment_encoded'] = enc.fit_transform(df[['Payment']])

df

col = 'Total'

df['log'] = np.log1p(df[col])
df['reciprocal'] = 1 / (df[col] + 1)
df['sqrt'] = np.sqrt(df[col])
df['square'] = np.square(df[col])

boxcox_trans, lambda_bc = stats.boxcox(df[col] + 1)
df['boxcox'] = boxcox_trans

pt = PowerTransformer(method='yeo-johnson')
df['yeojohnson'] = pt.fit_transform(df[[col]])

sm.qqplot(df['Total'], line='45')
plt.title("Original Total")
plt.show()

sm.qqplot(df['log'], line='45')
plt.title("Log Transformed")
plt.show()

sm.qqplot(df['boxcox'], line='45')
plt.title(f"Box-Cox Transformed (λ={lambda_bc:.2f})")
plt.show()

sm.qqplot(df['yeojohnson'], line='45')
plt.title("Yeo-Johnson Transformed")
plt.show()
```
# OUTPUT:

Label Encoder

<img width="1259" height="664" alt="image" src="https://github.com/user-attachments/assets/bc27b4c7-2934-4148-840c-8243c49a8967" />

OneHot Encoder

<img width="1244" height="463" alt="image" src="https://github.com/user-attachments/assets/fae5e4f6-847d-46cb-8e18-3c73703c2b0d" />

Ordinal Encoder 

<img width="1256" height="657" alt="image" src="https://github.com/user-attachments/assets/1cf6fd5e-b8d2-4cca-a210-08ec1b1abdac" />

Original Total Column 

<img width="764" height="574" alt="image" src="https://github.com/user-attachments/assets/a47b0dfd-35f8-4a05-a824-17ca33205e96" />

After Log Transformation

<img width="735" height="562" alt="image" src="https://github.com/user-attachments/assets/bf9fe63b-48cc-475a-a89a-ba3453697365" />

After Box-Cox Transformation

<img width="718" height="558" alt="image" src="https://github.com/user-attachments/assets/aff9d2e4-027a-42a7-b389-a6190fde6973" />

After Yeo-Johnson Transformed

<img width="747" height="564" alt="image" src="https://github.com/user-attachments/assets/98b77ad9-9c44-402e-bac5-10feaf6a41f5" />


# RESULT:
Hence, various encoders such as Label Encoder, OneHot Encoder and Ordinal Encoder. Feature transformations such as log transformation, Box-Cox transformation and Yeo-Johnson transfomation are performed.

       
