# set base image (host OS)
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN sh -c 'echo "APT { Get { AllowUnauthenticated \"1\"; }; };" > /etc/apt/apt.conf.d/99allow_unauth'
RUN apt -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update
RUN apt-get -y install wget
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/apt.conf.d/99allow_unauth cuda-keyring_1.0-1_all.deb
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80
RUN apt-get -y update && apt-get -y upgrade

# copy the content of the local src directory to the working directory
WORKDIR /usr/local/eden
COPY . .

# install the others
RUN ./setup.sh

# command to run on container start
ENTRYPOINT [ "python", "server.py", "--num-workers", "1", "--port", "5656" "--redis-host", "eden-stable-diffusion", "--redis-port", "6379" ]