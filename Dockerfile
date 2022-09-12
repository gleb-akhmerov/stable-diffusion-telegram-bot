FROM nvcr.io/nvidia/pytorch:22.04-py3
MAINTAINER Peter Willemsen <peter@codebuffet.co>

WORKDIR /opt/ldm_package

# "-c advice.detachedHead=false" is to supress detached head warning
RUN git clone --progress https://github.com/openai/CLIP && \
    cd CLIP && \
    git -c advice.detachedHead=false checkout d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
RUN git clone --progress https://github.com/CompVis/taming-transformers && \
    cd taming-transformers && \
    git -c advice.detachedHead=false checkout 24268930bf1dce879235a7fddd0b2355b84d7ea6

# for opencv
RUN apt-get update && \
    apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

COPY ./setup.py /opt/ldm_package
COPY ./ldm /opt/ldm_package/ldm
COPY ./configs /opt/ldm_package/configs
COPY environment.yaml /opt/ldm_package

# For the ldm-dev user
RUN chmod 777 -R /opt/ldm_package

WORKDIR /opt/ldm

# Add dev user
RUN useradd -ms /bin/bash ldm-dev && \
    usermod -aG sudo ldm-dev && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ldm-dev

RUN conda env create -f /opt/ldm_package/environment.yaml
RUN pip3 install -e /opt/ldm_package
RUN conda run -n ldm pip install pytorch-lightning==1.5
RUN conda run -n ldm pip install python-telegram-bot==13.13

# Create cache dir here to avoid problems with permissions
# with volume in docker-compose.yml
RUN mkdir -p /home/ldm-dev/.cache/huggingface

ENTRYPOINT ["conda", "run", "-n", "ldm", "--no-capture-output"]

CMD python optimized_txt2img_bot.py
