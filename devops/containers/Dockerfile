FROM jupyter/scipy-notebook

#all the code samples for the video series
VOLUME ["/src"]

#serve up a jupyter notebook 
WORKDIR /src
EXPOSE 8888

#this has security disabled which is less fuss for learning purposes
CMD jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True