FROM ubuntu:latest
MAINTAINER Nicole Meister "nmeister@princeton.edu"


# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app

# execute everyone's favorite pip command, pip install -r

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get install libgdal-dev -y
RUN apt-get install python-gdal -y
RUN apt-get install python-numpy python-scipy -y
RUN ls
RUN pip install geopandas
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# unblock port 80 for the Flask app to run on
EXPOSE 80

# execute the Flask app
CMD ["python", "app.py"]