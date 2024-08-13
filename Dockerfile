FROM jupyter/datascience-notebook:notebook-7.0.6

# USER root
# RUN apt-get update && apt-get install -y cmake
#
# USER jovyan
# RUN R -e 'install.packages(c("lme4","lmerTest","emmeans"), repos="http://cran.us.r-project.org")'
#
RUN pip install --upgrade pip

RUN pip install --upgrade torch

RUN pip install --upgrade wandb xgboost
#
# RUN pip install --upgrade concrete cmdstanpy arviz pymer4 statsmodels mlflow joypy scikit-learn krippendorff
#
# RUN python -c "from cmdstanpy import install_cmdstan; install_cmdstan(version='2.34.1')"
