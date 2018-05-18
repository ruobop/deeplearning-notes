# 深度学习笔记

## InsightFace-TF
### ckpt转pb
```python
# 载入图结构
new_saver = tf.train.import_meta_graph('model/InsightFace_iter_130000.ckpt.meta')

# 载入参数
new_saver.restore(sess, 'model/InsightFace_iter_130000.ckpt')

# 保存图结构为单独的文件
tf.train.write_graph(sess.graph_def, 'ckpt_to_pb', 'model.pb')

# 获得所有tensor name列表
tensor_name_list = [n.name for n in tf.get_default_graph().as_graph_def().node]

# 获得输入和输出tensor的名字
# 输入名字 = img_inputs，特征值输出名字 = resnet_v1_50_1/E_BN2/Identity

# 转换ckpt为pb
from tensorflow.python.tools import freeze_graph
freeze_graph('ckpt_to_pb/model.pb', '', False, 'model/InsightFace_iter_130000.ckpt', 'resnet_v1_50_1/E_BN2/Identity', '', '', 'ckpt_to_pb/freeze.pb', True, '')
