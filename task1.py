import pandas as pd
import numpy as np


data = {
    "ID": [1, 2, 3, 4, 5],
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Evan"],
    "Age": [21, 22, 23, 21, 22],
    "Grade": ["A", "B", "A", "C", "B"]
}

df = pd.DataFrame(data)
df_one_hot = pd.get_dummies(df['Name'], prefix='Name')
grade_mapping = {
    'A': 1,
    'B': 2,
    'C': 3
}
df['Grade_encoded'] = df['Grade'].map(grade_mapping)
df_final = pd.concat([df.drop(columns=['Name', 'Grade']), df_one_hot, df['Grade_encoded']], axis=1)

print("\nFinal DataFrame with Encodings:")
print(df_final)
final_array = df_final.to_numpy()
print("\nFinal Data as NumPy Array:")
print(final_array)
