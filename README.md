# 本仓库用于记录`TensorRT`的学习进度:alarm_clock:



**参考资料：**

1. [知乎](https://zhuanlan.zhihu.com/p/371239130)



**安装流程：**

1. 下载链接：[nvidia](https://developer.nvidia.com/nvidia-tensorrt-download)

2. 下载好之后进行解压

*这里需要注意一下`cuda`和`cudnn`的版本*

```shell
tar -zxvf TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz
```

3. 然后配置环境

```bash
export LD_LIBRARY_PATH=/home/admin1/TensorRT-8.2.5.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/admin1/TensorRT-8.2.5.1/lib::$LIBRARY_PATH
```

4. 更新环境

```shell
source ~/.bashrc
```

