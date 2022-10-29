###This is a code for leaning Two moon dataset using CBC Reject Options
import numpy as np
import sklearn as sk
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

################

k = 4
c = 2
eps = 0.01
samples = 500
train_iter = 5*samples
test_iter = 200
data_dim = 2
lambVal = 0.01
[prototypes, nonesense]= sk.datasets.make_moons(n_samples=k, shuffle=True, noise=0.1, random_state=0)
#plt.scatter(np.transpose(prototypes)[0], np.transpose(prototypes)[1], c='g')
pos_reasoning = 0.4* np.ones((c,k))
neg_reasoning = 0.4* np.ones((c,k))
neu_reasoning = np.ones((c,k)) - pos_reasoning - neg_reasoning

[x_train,y_train] = sk.datasets.make_moons(n_samples=samples, shuffle=True, noise=0.1, random_state=0)
[x_test,y_test] = sk.datasets.make_moons(n_samples=test_iter, shuffle=True, noise=0.1, random_state=0)

#here lies the functions

def sig(x,lam):
    return 1/(1+np.exp(-(x/lam)))

def phi(x,y,lam):
    return (1/lam) * sig(deltaP(x,y),lam) * (1-sig(deltaP(x,y),lam))

def deltaP(x, y_true):
        y_pred = predictorp(x)
        tem = y_pred
        s = np.argmax(y_pred)
        if s == y_true:
            tem[s] = 0
            s = np.argmax(tem)
        return y_pred[s] - y_pred[y_true]


def FirstAndSecond(x, y_true):
        y_pred = predictor(x)
        tem = y_pred
        s = np.argmax(y_pred)
        if s == y_true:
            tem[s] = 0
            s = np.argmax(tem)
        return s


def d(x,w):
    return np.exp(-2*(np.linalg.norm(x - w)) ** 2)

def Dd(x,w):
    return 4* d(x,w) * (x-w)

def predictor(x):
    prob = np.zeros(c)
    for j in range(c):
        ard = 0
        bird = 0
        cird = 0
        for z in range(k):
            ard = ard + d(x, prototypes[z]) * pos_reasoning[j,z]
            bird = bird + (1- d(x, prototypes[z,])) * neg_reasoning[j,z]
            cird = cird + (1 - neu_reasoning[j,z])
        prob[j] = (ard + bird)/cird
    return prob

def predictorp(x):
    vec = predictor(x)
    return vec/np.sum(vec)

def derivativePW(x,t,j):
    pr = predictor(x)
    N=np.sum(pr)
    return (1/N)**2 * derivativePhW(x,t,j) * N - derivaitveNW(x,t) * pr[j]

def derivaitveNW(x,t):
    temp = 0
    for j in range(c):
        temp = derivativePhW(x, t, j) + temp
    return temp

def derivativePhW(x,t,j):
    cird = 0
    for z in range(k):
        cird = cird + (1 - neu_reasoning[j,z])
    return (1/cird)*((pos_reasoning[j,t] - neg_reasoning[j,t])* Dd(x,prototypes[t]))

def derivativerp(x,j,t):
    cird = 0
    pr = predictor(x)
    N = np.sum(pr)
    for z in range(k):
        cird = cird + (1 - neu_reasoning[j, z])
    return (N-pr[j])*d(x,prototypes[t])/(cird * N**2)

def derivativern(x,j,t):
    cird = 0
    pr = predictor(x)
    N = np.sum(pr)
    for z in range(k):
        cird = cird + (1 - neu_reasoning[j, z])
    return (N - pr[j]) * (1-d(x, prototypes[t])) / (cird * N ** 2)

##training phase



for ii in range(train_iter):
    i= ii % 200
    arda = phi(x_train[i], y_train[i], lambVal)
    S = FirstAndSecond(x_train[i], y_train[i])

    for t in range(k):
        W_grad = arda * (derivativePW(x_train[i],t,S) - derivativePW(x_train[i],t,y_train[i]))
        prototypes[t] = prototypes[t] - eps * W_grad

    for t in range(k):
        tem = pos_reasoning[S,t] - eps * arda * derivativerp(x_train[i],S,t)
        if tem<1 and tem>0:
            pos_reasoning[S, t] = tem
        tem = pos_reasoning[y_train[i],t] + eps * arda * derivativerp(x_train[i], y_train[i],t)
        if tem<1 and tem>0:
            pos_reasoning[y_train[i], t] =tem
        tem = neg_reasoning[S,t] - eps * arda * derivativern(x_train[i],S,t)
        if tem<1 and tem>0:
            neg_reasoning[S, t] = tem
        tem =  neg_reasoning[y_train[i], t] + eps * arda * derivativern(x_train[i], y_train[i],t)
        if tem<1 and tem>0:
            neg_reasoning[y_train[i], t] =tem

    neu_reasoning = np.ones((c, k)) - pos_reasoning - neg_reasoning

##evaluation phase
count = 0
misclassamples = np.zeros((1,2))

for i in range(test_iter):
    ay = np.argmax(predictorp(x_test[i]))
    may = y_test[i]
    if ay == may:
        count = count +1
    if ay != may:
        vi = np.reshape(x_test[i], (1, 2))
        misclassamples = np.concatenate((misclassamples, vi))

print("accuarcy is ", count/test_iter)

tex = [1,2,3,4]
numer = np.array(range(k))
numer2 = np.array(range(3))
tex2 = ['r+','r-','r0']


fig = plt.figure(figsize=(8,6))

fig.add_subplot(4, 1, 1)
plt.scatter(np.transpose(x_test)[0], np.transpose(x_test)[1], c=y_test)
plt.scatter(np.transpose(prototypes)[0], np.transpose(prototypes)[1], c='r')
plt.scatter(np.transpose(misclassamples)[0], np.transpose(misclassamples)[1], c='black')
plt.scatter(0, 0, c='yellow')
for i, txt in enumerate(tex):
    plt.annotate(txt, (np.transpose(prototypes)[0,i], np.transpose(prototypes)[1,i]))


ax1 = fig.add_subplot(4, 1, 2)
img = np.stack((pos_reasoning[0], neg_reasoning[0], neu_reasoning[0]))
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.yticks(numer2, tex2)
plt.xticks(numer, tex)
ax1.title.set_text('Purple')

ax2 = fig.add_subplot(4, 1, 3)
img = np.stack((pos_reasoning[1], neg_reasoning[1], neu_reasoning[1]))
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.yticks(numer2, tex2)
plt.xticks(numer, tex)
ax2.title.set_text('Yellow')


shades = np.linspace(0,1,11)
im = np.reshape(shades, (1, 11))
fig.add_subplot(4, 1, 4)
plt.imshow(im, cmap='gray', vmin=0, vmax=1)
y = np.array(range(11))
shades = np.around(shades, 2)
plt.xticks(y, shades)
plt.yticks([])

plt.tight_layout()
plt.show()

print(pos_reasoning)
print(neg_reasoning)
print(neu_reasoning)