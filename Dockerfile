FROM continuumio/miniconda:latest

WORKDIR /home/forecaster-toolbox

COPY environment.yml ./
COPY data ./data
COPY forecaster_service.py ./
COPY model_training.py ./
COPY utils.py ./

RUN conda env create -f environment.yml
RUN echo "source activate forecaster_toolbox" > ~/.bashrc
ENV PATH /opt/conda/envs/forecaster_toolbox/bin:$PATH

EXPOSE 5000