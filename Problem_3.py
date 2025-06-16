import pandas as pd
import numpy as np

names = [f"Student{i}" for i in range(1, 11)]
subjects = np.random.choice(['Math', 'Science', 'English'], size=10)
scores = np.random.randint(50, 101, size=10)

grades = []
for score in scores:
    if score >= 90:
        grades.append('A')
    elif score >= 80:
        grades.append('B')
    elif score >= 70:
        grades.append('C')
    elif score >= 60:
        grades.append('D')
    else:
        grades.append('F')
df_students = pd.DataFrame({
    'Name': names,
    'Subject': subjects,
    'Score': scores,
    'Grade': grades
})

sorted_df = df_students.sort_values(by='Score', ascending=False)
print("Sorted by Score:\n", sorted_df)

avg_scores = df_students.groupby('Subject')['Score'].mean()
print("Average score per subject:\n", avg_scores)
def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

df_pass = pandas_filter_pass(df_students)
print("Students with Grade A or B:\n", df_pass)
