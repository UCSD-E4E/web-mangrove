FROM ubuntu:latest

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app


# Update base container install
RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install -y python3-pip python-dev build-essential

# Install GDAL dependencies
RUN apt-get install -y libgdal-dev locales

# Ensure locales configured correctly
RUN locale-gen en_US.UTF-8
ENV LC_ALL='en_US.utf8'

# Set python aliases for python3
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc

RUN apt-get install software-properties-common -y
RUN apt-get update -y

RUN add-apt-repository ppa:ubuntugis/ppa

RUN apt-get update -y
RUN apt-get install gdal-bin -y

RUN ogrinfo --version

RUN apt-get install libgdal-dev

# test
RUN gdal-config --version

# This will install latest version of GDAL
# RUN pip3 install GDAL==2.4.2

# do i need this??
# RUN apt-get install python-gdal -y
RUN apt-get install python-numpy python-scipy -y

# execute pip install -r
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt


# unblock port 80 for the Flask app to run on
EXPOSE 80

# execute the Flask app
CMD ["python", "app.py"]