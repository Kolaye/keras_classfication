# encoding: utf-8
import requests
import argparse

#这是本地调用云端模型的程序

# Initialize the keras REST API endpoint URL.
REST_API_URL = 'http://127.0.0.1:5001/predict'
import time 

def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        for (i, result) in enumerate(r['predictions']):
            print(result)

    # Otherwise, the request failed.
    else:
        print('Request failed')


if __name__ == '__main__':


    t1 = time.time()
    img_path = './test_image/1.jpg'
    predict_result(img_path)
    t2 = time.time()
    print(t2-t1)
