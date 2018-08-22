FROM continuumio/miniconda3
LABEL maintainer="Fabian Witt <fabian.witt@t-online.de>"

# for proxy usage
# ENV http_proxy=
# ENV https_proxy=

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

RUN ["mkdir", "notebooks"]

ADD environment.yml environment.yml

RUN /opt/conda/bin/conda env update -n base -f environment.yml && \
    /opt/conda/bin/conda clean -ayq

COPY conf/.jupyter /root/.jupyter
COPY run_jupyter.sh /

RUN ["chmod", "+x", "/run_jupyter.sh"]

# Jupyter port
EXPOSE 8888

# Store notebooks in this mounted directory
VOLUME /notebooks

CMD ["/run_jupyter.sh"]

# for proxy usage
# ENV http_proxy ""
# ENV https_proxy ""
