# 记录从源码编译tf c++接口

### 下载源代码
```git clone https://github.com/tensorflow/tensorflow.git```

### 安装bazel
```echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel```
