FROM nvcr.io/nvidia/pytorch:22.04-py3
MAINTAINER Peter Willemsen <peter@codebuffet.co>

RUN conda

RUN mkdir -p /opt/ldm_package
COPY ./setup.py /opt/ldm_package
COPY ./ldm /opt/ldm_package/ldm
COPY ./configs /opt/ldm_package/configs
COPY ./optimizedSD /opt/ldm_package/optimizedSD
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

ENTRYPOINT ["conda", "run", "-n", "ldm", "--no-capture-output"]
