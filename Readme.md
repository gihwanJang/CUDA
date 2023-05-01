# CUDA

## CUDA 실습 Repository 입니다.

### 개발 환경

- OS : Ubuntu
    - Version : 18.04
- IDE : vscode 
    - Version: 1.76.1
- Compiler : GCC
    - Version: 7.5.0
- Language : C++, Make, CUDA

- hardware
    - cpu : Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
    - gpu  
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
|  0%   54C    P5    18W / 160W |    552MiB /  6144MiB |     33%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1412      G   /usr/lib/xorg/Xorg                243MiB |
|    0   N/A  N/A      2864      G   /usr/bin/gnome-shell               45MiB |
|    0   N/A  N/A      3547      G   ...RendererForSitePerProcess      139MiB |
|    0   N/A  N/A      4411      G   ...473269103524995576,131072      120MiB |
+-----------------------------------------------------------------------------+