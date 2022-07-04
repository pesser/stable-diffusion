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
RUN apt-get install -y git wget curl libgl1-mesa-glx libglib2.0-0

# copy the content of the local src directory to the working directory
WORKDIR /usr/local/eden
COPY . .

# requirements for stable-diffusion
RUN pip install -r requirements.txt
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install eden-python

# setting up client
RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc
RUN chmod +x mc
RUN --mount=type=secret,id=BUCKET_URL \
    --mount=type=secret,id=BUCKET_USER \
    --mount=type=secret,id=BUCKET_PASSWORD \
    export BUCKET_URL=$(cat /run/secrets/BUCKET_URL) && \
    export BUCKET_USER=$(cat /run/secrets/BUCKET_USER) && \
    export BUCKET_PASSWORD=$(cat /run/secrets/BUCKET_PASSWORD) && \
    ./mc alias set eden $BUCKET_URL $BUCKET_USER $BUCKET_PASSWORD

# download model (previously #RUN gdown {model_link} -O {ckpt})
RUN ./mc cp eden/models/test/test .

ENTRYPOINT ["python", "server.py", "--num-workers", "1", "--port", "5656" "--redis-host", "eden-diffusion-redis", "--redis-port", "6379"]
