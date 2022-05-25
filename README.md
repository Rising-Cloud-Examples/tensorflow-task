# tensorflow-task
This guide will walk you through the simple steps needed to build and run an Image Processing app using TensorFlow on Rising Cloud.  

If you'd prefer to watch a video, check out our YouTube of the tutorial.

[![Video Player](https://cms.risingcloud.com/uploads/Video_Player_b69c4aa4ff.png)](https://youtu.be/BfK_oOAX0BA)

# 1. Install the Rising Cloud Command Line Interface (CLI)
In order to run the Rising Cloud commands in this guide, you will need to [install](https://risingcloud.com/docs/install) the Rising Cloud Command Line Interface. This program provides you with the utilities to setup your Rising Cloud Task or Web Service, upload your application to Rising Cloud, setup authentication, and more.

# 2. Login to Rising Cloud Using the CLI
Using a command line console (called terminal on Mac OS X and command prompt on Windows) run the Rising Cloud login command. The interface will request your Rising Cloud email address and password.

```risingcloud login```

# 3. Initialize Your Rising Cloud Task
Create a new directory on your workstation to place your project files in, then open this directory with your command line.

Using the command line in your project directory, run the following command replacing $TASK with your unique task name.  Your unique task name must be at least 12 characters long and consist of only alphanumeric characters and hyphens (-). This task name is unique to all tasks on Rising Cloud. A unique URL will be provided to you for sending jobs to your task. If a task name is not available, the CLI will return with an error so you can try again.

```risingcloud init -s $TASK_URL```

This creates a risingcloud.yaml file in your project directory. This file will be used to configure your build script.

# 4. Create your Rising Cloud Task

**Configuring your I/O**

When a Rising Cloud Job is run, input is written to request.json in the top level of your project directory. Your application will need to read this to respond to it. When your application terminates, output is read from response.json, if it exists, and is stored in Rising Cloud’s Job Results database for retrieval.

Input to Rising Cloud Tasks has to come a JSON. If you are planning on using pdfs, images, or other non-JSONable data as input to your neural net, you will have to use the input JSON to give your application URLs to download the data from. Likewise, if the output of your application is an image, you will need your application to store the application in a database and return information on how to retrieve it in the output (such as a URL to an Amazon S3 object.) See our [Statelessness](https://risingcloud.com/docs/statelessness) guide to understand why this is, and for information about connecting to external data sources.

In this example, we will have our users specify a URL to download an image from.

**Create Your TensorFlow Program**

For this example, we will use a pretrained Keras model to run predictions on user input. In your project directory, create a file named Classifier.py, and in it, write the following contents:

```
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import json
import requests
import os
#Load model with pretrained weights (from keras.io)
"""
NOTE: the following line downloads the model from keras each time the program is run.
In practice, you should load the pretrained model from your project directory.
"""
model = ResNet50(weights='imagenet')
img_path = 'img.jpg'
request = {}
#Read Rising Cloud Task input
with open("request.json", 'r') as f:
    request = json.load(f) 
    f.close()
    ret = []
#Downloads file from “img_url” field in request.json to img.jpg locally
with open(img_path, 'wb+') as f:
    response = requests.get(request["img_url"], stream=True)
    if not response.ok:
        print (response)
    for block in response.iter_content(1024):
        if not block:
            break
        f.write(block)
#runs the image through the classifier
with open(img_path, "rb") as f:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]
    ret = [p[1] for p in top_preds]
response = {"guesses": ret}
#puts top 3 guesses in response.json
with open("response.json", 'w+') as f:
    json.dump(response, f)
#Cleanup: deletes img.jpg
os.remove(img_path)

```

This Python script will load the weights of the pretrained Keras model, download the image from the “img_url” field in the JSON body of job requests, and return its three best guesses about what the image is.

**TIP:** You can load the model via a daemon to avoid loading the model each time a worker receives a job, but then you will have to find a way to pass data to and from the daemon.

**Configure your risingcloud.yaml**

When you ran "risingcloud init", a new risingcloud.yaml file should have generated in your project directory. Open that file now in your editor.  In dependencies, we need to install pip and use it to install Pillow, TensorFlow, and requests. Change deps to:

```
deps: 
- apt update && apt upgrade -y
- apt install -y python3-pip
- pip3 install requests
- pip3 install Pillow
- pip3 install tensorflow
```

We need to tell Rising Cloud what to run when a new request comes in.  Change run to:

```run: python3 Classifier.py```

**TIP:** Freezing your Python application can make your image smaller and result in more efficient scaling. Check out our Python [Freezing](https://risingcloud.com/docs/freezing-your-application) guide for more details on this.

# 5. Build and Deploy your Rising Cloud Task
Use the push command to push your updated risingcloud.yaml to your Task on Rising Cloud.

```risingcloud push```

Use the build command to zip, upload, and build your app on Rising Cloud.

```risingcloud build```

Use the deploy command to deploy your app as soon as the build is complete.  Change $TASK to your unique task name.

```risingcloud deploy $TASK```

Alternatively, you could also use a combination to push, build and deploy all at once.

```risingcloud build -r -d```

You can set environment variables for your app either from a file, your risingcloud.yaml, or each variable one by one.

```
risingcloud setenv —file /path/to/.env
risingcloud setenv —var $NAME=$VALUE
```

Rising Cloud will now build out the infrastructure necessary to run and scale your application including networking, load balancing and DNS.  Allow DNS a few minutes to propogate and then your app will be ready and available to use!

# 6. Queue Jobs for your Rising Cloud Task

**Make requests**

Rising Cloud will take some time to build and deploy your Rising Cloud Task. Once it is done, you can make HTTPS POST requests with JSON bodies to https://{your project URL}.risingcloud.app/risingcloud/jobs to queue jobs for Rising Cloud Task. These requests will return JSON responses with a “jobId” field containing the ID of your job. Make an HTTP GET request to https://{your project URL}.risingcloud.app/risingcloud/jobs/{job ID} in order to check on the status of the job. If the response’s “status” field is “Completed”, the result of the job will appear under the “result” field in the JSON object.

Making a request with a JSON body of:

```{"img_url": "https://farm3.staticflickr.com/2453/3693487960_e6b276918d_z.jpg"}```

(This URL is to a picture of a zebra from the COCO dataset.)

should cause the “result” field in a completed Job Status to be:

```
	"result": {
		"guesses": [
			"zebra",
			"prairie_chicken",
			"hartebeest"
		]
```

Congratulations, you’ve successfully used TensorFlow on Rising Cloud!
