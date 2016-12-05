import pandas as pd

fram = pd.read_csv("framingham.csv")

theta = [1,1]
X = [[0, 1],[0,0]]
#print(h(theta,X))
small = fram.iloc[:10,:2].values
#print(small)
theta[1] = 0.02
#print(h(theta,small))

fram.dropna(inplace=True)  #drop nulls
#print(fram.shape)

x = np.ones((100,3))
x[:,1:] = normalize(fram[['age','glucose']].values[:100])
#print(x[:10])
y = fram.TenYearCHD[:100]
#print(logfit(x,y))
th = (logfit(x,y,0.1,1000))
print(th)
#print(h(th,x))

#plt.hist(h(th,x))
#plt.show()

P = h(th,x)
print(((P > 0.2) == y).mean())
print(pd.crosstab(P > 0.2, y))

#true positive ratio
TPR = 11/18

FPR = 33/82

print(pd.crosstab(P > 1, y))



si = np.argsort(P)[::-1] #index of the sort
Ps = P[si]
Ys = y[si]
print(Ps,Ys[:10])
print(y[:10].sum()/y.sum())

print(y[:10].sum()/y.sum())


tpr,fpr = tprfpr(P, y)
# plt.scatter(tpr,fpr)
# plt.plot(tpr,fpr)
# plt.show()

print(auc(fpr,tpr))