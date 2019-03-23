import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
%matplotlib inline
from PIL import Image
import time

Path_Plan='CAD-60-Images/XY'
def load_Dataset(Nom_Plan):
    
    TRAIN_DIR =Nom_Plan
    filesName = os.listdir(TRAIN_DIR)
    train_image_file_names = [];
    for i in range(len(filesName)):
        train_image_file_names.append(TRAIN_DIR+"/"+filesName[i])
    return(train_image_file_names)
list=[]
i=0
while (i<104):
    r=np.random.randint(1,1800)
    if r not in list: 
        list.append(r)
        i+=1
def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels,C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot
def Y_all():
    nombre_classe=17 # nombre de geste 
    nombre_totatle_images=1904 # nombre d'image totale de la base de données 112*17(112 images pour chaque geste)
    nombre_image_geste=112 # nombre d'image pour chaque geste 28*4 (28 images pour chaque personne)
    labels = np.arange(0,nombre_classe)
    one_hot = one_hot_matrix(labels, C = nombre_classe)
    Y_all=np.zeros(0)

    for i in range(nombre_classe):
        for j in range(0,nombre_image_geste):
            #Y_all.append(one_hot[i])
            Y_all=np.append(Y_all,one_hot[i])

    Y_all=np.reshape(Y_all,(nombre_totatle_images,nombre_classe)) 
    return Y_all
def Data_set(load_Data,load_Label):
    A=load_Data
    B=load_Label
    nombre_image_test=104
    nombre_image_train=1800
    nombre_classe=17

    X_train=np.zeros(0)
    X_test=np.zeros(0)
    Y_train=np.zeros(0)
    Y_test=np.zeros(0)

    for i in range(len(list)):
        X_test=np.append(X_test,A[list[i]])
        Y_test=np.append(Y_test,B[list[i]])
    Y_test = np.reshape(Y_test, (nombre_image_test,nombre_classe,1))

    for i in range(len(A)):
        if A[i] not in X_test:                       
                    X_train = np.append(X_train,A[i])

    for i in range(len(B)):
        if i not in list:
            Y_train = np.append(Y_train,B[i])
        else:
            pass
    Y_train=np.reshape(Y_train,(nombre_image_train,nombre_classe,-1))
    return (X_train,Y_train,X_test,Y_test)
def read_image(X):    
    im=np.zeros(0)
    A=load_Dataset(Path_Plan)
    for i in range(len(X)):
        #print("size of image",im.size)
        img=Image.open(X[i], "r")
        comp = Image.new("RGB", (28,28), (255, 255, 255, 0))  # 0 for transparency
        comp.paste(img, (0, 0))
        im =np.append(im,comp)
    
    return im
A=load_Dataset(Path_Plan)
B=Y_all()
[X1,Y_train,X2,Y_test]=Data_set(A,B)
inputs=28
nombre_image_test=104
nombre_image_train=1800
nombre_classe=17
X_1=read_image(X1)
X_train=np.reshape(X_1,(nombre_image_train,inputs,inputs,3))
X_2=read_image(X2)
X_test=np.reshape(X_2,(nombre_image_test,inputs,inputs,3))
Y_test=np.reshape(Y_test,(nombre_image_test,nombre_classe),-1)
Y_train=np.reshape(Y_train,(nombre_image_train,nombre_classe),-1)
tf.reset_default_graph()
n_inputs = 28 # number of input vector elements i.e. pixels per training example
n_classes = 17 # number of classes to be classified
learning_rate = 0.001
training_iters = 50
batch_size = 100
# input and output vector placeholders
x = tf.placeholder(tf.float32, [None, n_inputs,n_inputs,3])
y = tf.placeholder(tf.float32, [None, n_classes])

# fully connected layer
fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)
weights = {
    'wc1': tf.get_variable('W-0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W-1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W-2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W-3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W-6', shape=(128,17), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B-0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B-1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B-2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B-3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B-4', shape=(17), initializer=tf.contrib.layers.xavier_initializer()),
}
def conv_net(img, weights, biases):  
    img = tf.reshape(img, [-1, 28, 28, 3])
    # 1st convolutional layer
    conv1 = tf.nn.conv2d(img, weights["wc1"], strides=[1, 4, 4, 1], padding="SAME", name="conv1")

    conv1 = tf.nn.bias_add(conv1, biases["bc1"])
    conv1 = tf.nn.relu(conv1)
    
    conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 2nd convolutional layer
    conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 1, 1, 1], padding="SAME", name="conv2")
    conv2 = tf.nn.bias_add(conv2, biases["bc2"])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 3rd convolutional layer
    conv3 = tf.nn.conv2d(conv2, weights["wc3"], strides=[1, 1, 1, 1], padding="SAME", name="conv3")
    conv3 = tf.nn.bias_add(conv3, biases["bc3"])
    conv3 = tf.nn.relu(conv3)
    
    
    #flatten 
    conv3Flatten = tf.contrib.layers.flatten(conv3)
    fc3=tf.contrib.layers.fully_connected(conv3Flatten,17,activation_fn=None)
    fc3 = tf.nn.softmax(fc3)
        # stretching out the 5th convolutional layer into a long n-dimensional tensor
    #shape = [-1, weights['wd1'].get_shape().as_list()[0]]
    #flatten = tf.reshape(conv3, shape)

        # Fully connected layer
    # 3rd fully connected layer
    #fc3 = fc_layer(flatten, weights["out"], biases["out"], name="out")
    #fc3 = tf.nn.softmax(fc3)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out=fc3
    return out
pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
#all_saver = tf.train.Saver()
init = tf.global_variables_initializer()
def time_execution(dur):
    if(dur>3600):
        heurs=int(dur/3600)
        dur=dur%3600
        print(heurs,"heurs")
    if(dur>60):
        minut=int(dur/60)
        dur=dur%60
        print(minut,"minut",int(dur),"seconde")
with tf.Session() as sess:
    sess.run(init) 
    #all_saver.save(sess, 'Dataset_Tensorflow' + '/data-all')
    
    t1 = time.clock();
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)

    for i in range(training_iters):
        opt = sess.run(optimizer, feed_dict={x: X_train,y: Y_train})
        loss, acc = sess.run([cost, accuracy], feed_dict={x: X_train,
                                                                         y: Y_train})
        #print("Iter " + str(i) + ", Loss= " + \
              #"{:.6f}".format(loss) + ", Training Accuracy= " + \
              #"{:.5f}".format(acc))
        #print("Optimization Finished!")

        #print("wc01="+str(weights["wc01"].eval()[1,1,1]))
        
        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: X_test,y : Y_test})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        #print("Testing Accuracy:","{:.5f}".format(test_acc))
    dur=time.clock()-t1
    time_execution(dur)
    summary_writer.close()

