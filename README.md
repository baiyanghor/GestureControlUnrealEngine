# Gesture Control Unreal Engine
This repository demos how to integrate Google MediaPipe hands gesture module with Unreal Engine

# The environments I use:
  Python 3.11
  Google Mediapipe
  Unreal Engine 5.3
  Chaos Cloth Plugin

# How to use:
  With the Unreal project running
  cd you\local\path\PythonScripts
  Run:
  python.exe GestureToUE

# Notes:
I developed the TCP communication module with reference on Unreal Forum
the IP and port hard coded in 127.0.0.1 port: 27015

# Check the demo video here
  https://youtu.be/8T3Aph6KUPo

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
