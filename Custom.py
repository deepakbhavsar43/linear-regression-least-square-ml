import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv("data.csv")
# print(f"{X} \n\n {Y}")
data = pd.DataFrame({
    "X": [1,2,3,4,5],
    "Y": [3,4,2,4,5]
})
X = data["X"]
Y = data["Y"]


# calculate mean of X and Y
data_length = len(X)
xmean, ymean = 0, 0
for i in range(0, data_length):
    xmean += X[i]
    ymean += Y[i]
xmean = xmean / data_length
ymean = ymean / data_length

# calculate value of m
numerator, denominator = 0, 0
for i in range(0, data_length):
    numerator += (X[i] - xmean) * (Y[i] - ymean)
    denominator += (X[i] - xmean) ** 2
m = numerator / denominator

# Calculate C
c = ymean - m * xmean

# Predict value of Y
YPredicted = []
for i in range(0, data_length):
    temp = m * X[i] + c
    YPredicted.append(temp)

# Claculate R Square
top, down = 0, 0
for i in range(0, data_length):
    top += (Y[i] - YPredicted[i]) ** 2
    # top += (YPredicted[i] - ymean) ** 2  # 0.3076923076923078
    down += (Y[i] - ymean) ** 2
r2Score = 1 - (top / down)
print("Predictions: ",YPredicted)
print("Goodness of Fit: ",r2Score)

# Plot graph
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.scatter(X, Y, label="Scatter Plot")
plt.plot([X.min(), X.max()], [Y.min(), Y.max()], color="#ef5423", label="Regression Line")
plt.legend()
plt.show()
