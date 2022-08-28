from pyexpat import model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

#导入数据集
digits=load_digits()
X=digits.data.astype(np.float32)
y=digits.target.astype(np.float32).reshape(-1,1)
#对数据集进行归一化
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X=X_scaled.reshape(-1,8,8,1)
#拆分数据集
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)
#对测试集的目标数据进行独热编码
one_hot_encoder=OneHotEncoder()
one_hot_encoder.fit(y)
y_train_coded=one_hot_encoder.transform(y_train).todense()
y_test_coded=one_hot_encoder.transform(y_test).todense()
#产生批次
batch_size=4
def my_batch(X,y,n_samples,batch_size):
    for batch_i in range (n_samples//batch_size):
        start=batch_i*batch_size
        end=start+batch_size
        batch_xs=X[start:end]
        batch_ys=y[start:end]
        yield batch_xs,batch_ys
tf.reset_default_graph()
#创建神经网络输入层，先以占位符的形式创建，之后进行赋值
tf_X=tf.placeholder(tf.float32,[None,8,8,1])
tf_y=tf.placeholder(tf.float32,[None,10])
#卷积层1：滤波、偏置、Relu激励
conv1_w=tf.Variable(tf.random_normal([3,3,1,10]))
conv1_b=tf.Variable(tf.random_normal([10]))
conv1_out=tf.nn.relu(tf.nn.conv2d(tf_X,conv1_w,strides=[1,1,1,1],padding='SAME')+conv1_b)
#池化层1：对卷积层1的输出进行池化
pool1_out=tf.nn.relu(tf.nn.max_pool(conv1_out,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME'))
#print("池化层1结果为：",pool1_out)
#卷积层2：滤波、偏置、Relu激励
conv2_w=tf.Variable(tf.random_normal([3,3,10,5]))
conv2_b=tf.Variable(tf.random_normal([5]))
conv2_out=tf.nn.relu(tf.nn.conv2d(pool1_out,conv2_w,strides=[1,2,2,1],padding='SAME'))
#池化层2：对卷积层2的输出进行池化
pool2_out=tf.nn.relu(tf.nn.max_pool(conv2_out,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME'))
#print("池化层2结果为：",pool2_out)
#归一化
batch_mean,batch_var=tf.nn.moments(pool2_out,[0,1,2],keep_dims=True)
offset=tf.Variable(tf.zeros([5]))
scale=tf.Variable(tf.ones([5]))
epsilon=1e-3
bn_out=tf.nn.batch_normalization(pool2_out,mean=batch_mean,variance=batch_var,offset=offset,scale=scale,variance_epsilon=epsilon)
#print("归一化结果为：",bn_out)
#构建全连接层，全连接层有2个隐层，每个隐层含有50个神经元
bn_flat=tf.reshape(bn_out,[-1,2*2*5])#将特征图张成一维
fc1_w=tf.Variable(tf.random_normal([2*2*5,40]))
fc1_b=tf.Variable(tf.random_normal([40]))
fc1_out=tf.nn.relu(tf.matmul(bn_flat,fc1_w)+fc1_b)
fc2_w=tf.Variable(tf.random_normal([40,40]))
fc2_b=tf.Variable(tf.random_normal([40]))
fc2_out=tf.nn.relu(tf.matmul(fc1_out,fc2_w)+fc2_b)
#print("全连接层输出为：",fc2_out)
#输出层
out_w=tf.Variable(tf.random_normal([40,10]))
out_b=tf.Variable(tf.random_normal([10]))
pred=tf.nn.softmax(tf.nn.relu(tf.matmul(fc2_out,out_w)+out_b))
#定义并优化损失函数
loss=tf.nn.softmax_cross_entropy_with_logits(labels=tf_y,logits=pred)
train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
#计算模型准确率
y_pred=tf.math.argmax(pred,1)
bool_pred=tf.equal(tf.math.argmax(tf_y,1),y_pred)
accuracy=tf.reduce_mean(tf.cast(bool_pred,dtype=tf.float32))
#生成可保存的Saver
#network=tf.train.Saver(max_to_keep=1)
#训练模型
print("----------训练中----------")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res_last=0
    for epoch in range(1000):
        for batch_xs,batch_ys in my_batch(X_train,y_train_coded,len(y_train_coded),batch_size):
            sess.run(train_step,feed_dict={tf_X:batch_xs,tf_y:batch_ys})
        if(epoch%100==0):
            res=sess.run(accuracy,feed_dict={tf_X:X_train,tf_y:y_train_coded})
            print("第",epoch,"次迭代准确率为",res)
            '''if((res<=res_last) & (res_last-res>=0.001)):
                print("----------训练提前完成----------")
                break
            else:
                res_last=res'''
    print("----------训练完成----------")
    #保存模型
    #network.save(sess,"F:\python_code\models\digist_cnn_model.ckpt")
    #用完成训练的模型对测试集进行预测
    result_pred=y_pred.eval(feed_dict={tf_X:X_test}).flatten()   
#输出测试集预测结果并对模型进行评价 
print("测试集预测结果为：\n",result_pred)
print("测试集预测结果评价报告：\n",classification_report(y_test,result_pred.reshape(-1,1)))