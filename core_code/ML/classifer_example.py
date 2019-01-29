# 1 load the libraries needed and initialize the computational graph
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf

sess = tf.Session()
train_indices = np.random.choice(5, 4, replace=False)
test_indices = np.array(list(set(range(5)) - set(train_indices)))
# 2 Load the data and transform it to binary
iris = datasets.load_iris()
binary_target = np.array([1. if x == 0 else 0. for x in iris.
                         target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

# 3 Declaring batch size, data placeholders, and model variables
batch_size = 20
# Placeholder values are assigned during the training
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)  # float32 decrease the time needed
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
# 4 Declaring the model
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

# 5  Define the loss function

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

# 6 Telling TF how to optimize the loss by declaring optimizer function
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# 7 After we finish, every thing we should tell TF to initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# 8 Training with 1000 iterations = # of batches (It should)
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])  # petal length data input
    rand_x2 = np.array([[x[1]] for x in rand_x])  # petal width data input
    rand_y = np.array([[y] for y in binary_target[rand_index]])  # Targets
    # we Send all the placeholders
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data:
        rand_x2, y_target: rand_y})
    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))

# 9 Extract Model parameters and draw it
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
    ablineValues.append(slope * i + intercept)

setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa''')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()

