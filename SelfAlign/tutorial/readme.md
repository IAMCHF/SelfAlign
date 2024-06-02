![对齐流程](/media/hao/Sata500g/BaiduSyncdisk/Beijing/ISBRA-2024/SelfAlign/SelfAlign/tutorial/img/对齐流程.png)

tensorflow-2.12.0	python3.8-3.11	GCC 9.3.1	cuda11.8  cudnn8.6

**cudnn install(ubuntu22.04 deb install):**

sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb

sudo cp /var/cudnn-local-repo-ubuntu2204-8.6.0.163/cudnn-local-FAED14DD-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8

sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8

sudo apt-get install libcudnn8-samples=8.6.0.163-1+cuda11.8

**cudnn test:**

