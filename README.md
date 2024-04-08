# GestureControlUnrealEngine
 Powered by Google Mediapipe

Python 端环境需求：

> Python 3.11
> Google Mediapipe 深度学习套件: https://developers.google.com/mediapipe

Ureal Engine 端环境需求：

>UE 5.3.2
>Metahuman 插件
>Chaos Cloth 相关插件

****************************************
运行时先运行游戏，然后用如下命令启动Python 脚本

cd (自己电脑的下载目录)\PythonScripts
python.exe GestureToUE.py

**************************************
Socket 通讯用的是TCP协议， IP： 127.0.0.1 ，端口： 27015 ，如果没有其他应用占用端口可以直接运行。
如果要使用其他端口需要在代码中改一下，Python 端和UE 端要一致。纯技术 Demo，暂时没有做方便的用户配置界面。
