from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.ensemble import RandomForestClassifier
plt.switch_backend('TkAgg')


# 1 加载数据集
digits = load_digits()      # 手写数字的数据集
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(8, 6), dpi=200)
for index, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2, 4, index+1)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')  # 对图像进行处理
    plt.title('Digit:%i' % label, fontsize=20)
print("shape of raw image_data:{0}".format(digits.images.shape))
print("shape if data:{0}".format(digits.data.shape))
plt.show()

# 2 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=2)

# 3 训练模型
# 求出逻辑回归 Logistic 的精确度得分
clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=5000, random_state=42)
clf.fit(X_train, y_train)
print(clf.fit(X_train, y_train))

# 4 测试模型
accuracy = clf.score(X_test, y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__, accuracy))

# 接把测试数据集⾥的部分图⽚显⽰出来，并且在图⽚的左下⾓显⽰预测值，右下⾓显⽰真实值。
y_pred=clf.predict(X_test)
fig, axes=plt.subplots(4, 4, figsize=(8,8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.05, 0.05, str(y_pred[i]), fontsize=32, transform=ax.transAxes, color='green' if y_pred[i] == y_test[i] else 'red')
    ax.text(0.8, 0.05, str(y_test[i]), fontsize=32, transform=ax.transAxes, color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# 5 模型保存与加载
joblib.dump(clf, 'digits_svm.pkl')
print(joblib.dump(clf, 'digits_svm.pkl'))

clf2 = joblib.load('digits_svm.pkl')
clf2.score(X_test, y_test)
print(clf2.score(X_test, y_test))

# 模型的轻松更改
# 通过随机森林分类器RandomForestClassifier轻松替换逻辑归回LogisticRegression分类器
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
y_pred=clf.predict(X_test)
fig, axes=plt.subplots(5, 5, figsize=(8,8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.05, 0.05, str(y_pred[i]), fontsize=32, transform=ax.transAxes, color='green' if y_pred[i] == y_test[i] else 'red')
    ax.text(0.8, 0.05, str(y_test[i]), fontsize=32, transform=ax.transAxes, color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__, accuracy))


# 测试不同比例的得分-逻辑回归
testsize = 0.05
for i in range (0, 19):
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=testsize, random_state=2)
    # 求出逻辑回归 Logistic 的精确度得分
    clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(testsize)
    print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__, accuracy))
    testsize = testsize + 0.05
