# 背景
在学习CS231N时，线性分类器用到了SVM Loss，所以打算这里推导一样，并解释一下CS231N对SVM Loss的native实现和向量化实现
# 推导
给出SVM Loss的公式$L_i = \sum_{j\neq y_i}^{classnum} max(0, s_j-s_{y_i}+\delta) = \sum_{j\neq y_i}^{classnum}max(0, w_j*x_i^T - w_{y_i}*x_i^T+\delta)$，这个公式表达的意思就是svm算法是对每一个样本计算所有的不正确分类和正确分类之间的差距，如果差距大于delta就代表差距大，需要继续优化，所以对于每个样本把这些差距家取来，让差距和最小(损失函数)，这就是svm的思想。对于Loss的计算直接按照上面的公式即可，对于梯度的话，由于这里是一个max门，如果差距小于delta，代表max门的输出为0，这个时候梯度也是为0的，所以主要是要看当$w_j*x_i^T-w_{y_i}*x_i^T+\delta>0$的时候，梯度是什么样子，这里只有$w_j和w_{y_i}$两个参数，所以分别对这两个参数求导，可得梯度：

$j\neq y_i, w_j*x_i^T-w_{y_i}*x_i^T+\delta>0$ => $\frac{\partial L_i}{\partial w_j}=x_i^T$

$j\neq y_i, w_j*x_i^T-w_{y_i}*x_i^T+\delta>0$ => $\frac{\partial L_i}{\partial w_{y_i}}=-x_i^T$

$j\neq y_i, w_j*x_i^T-w_{y_i}*x_i^T+\delta<=0$ => $\frac{\partial L_i}{\partial w_{y_i}}=0$

$j\neq y_i, w_j*x_i^T-w_{y_i}*x_i^T+\delta<=0$ => $\frac{\partial L_i}{\partial w_j}=0$

# 代码实现
第一种是按照上诉推导直接计算的：

```
def svm_loss_native(W, X, y, reg):
  '''
  svm_loss的朴素实现
  输入的维度是D，有C个分类类别，并且我们在有N个例子的batch上进行操作
    输入:
    - W: 一个numpy array，形状是(D, C)，代表权重
    - X: 一个形状为(N, D)为numpy array，代表输入数据
    - y: 一个形状为(N,)的numpy array，代表类别标签
    - reg: (float)正则化参数
    f返回:
    - 一个浮点数代表Loss
    - 和W形状一样的梯度
  '''
  dW = np.zeros_like(W) #初始化权值矩阵
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 #note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i].T
        dW[:, y[i]] += -X[i].T
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return loss, dW
```
第二种是利用了python中numpy的向量化计算，提高了计算速度，这里需要解释一下，首先计算loss的时候先另外声明一个正确矩阵correct_class_scores，这个矩阵首先在每个样本对应的正确类别处对应的元素是1，然后把这个矩阵reshape成[N,1]的矩阵，然后做差的时候根据numpy的broadcast属性会自动延展成[样本数，类别数】的矩阵。在求梯度时，公式是$x_i^T$，加的次数为j，j就是没有正确分类的情况，这种情况有多少个呢？其实就是把每一行之前求取？的loss有效的结果先置1，再加起来的个数，由于每次修正错误列的时候还会修正一次正确类别的列，所以直接把这个加和结果取相反数就是正确类别的导数，最后再乘以X的转置，加L2正则化梯度即可。具体看下面的代码：

```
def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1) #[N, 1]
  margins = np.maximum(0, scores - correct_class_scores + 1)
  margins[range(num_train), list(y)] = 0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

  coffe_mat = np.zeros((num_train, num_classes))
  coffe_mat[margins > 0] = 1
  coffe_mat[range(num_train), list(y)] = 0
  coffe_mat[range(num_train), list(y)] = -np.sum(coffe_mat, axis=1)

  dW = (X.T).dot(coffe_mat)
  dW = dW/num_train + reg*W
  return loss, dW
```