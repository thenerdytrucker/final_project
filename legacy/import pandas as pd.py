import pandas as pd

df = pd.read_csv(
    'C:\\Users\\thene\\OneDrive\\Desktop\\archive\\lung_cancer.csv')

df.head()
df.dtypes
df.info()

test = "   this is a test string   "

print(test.strip())  # Output: "this is a test string"
