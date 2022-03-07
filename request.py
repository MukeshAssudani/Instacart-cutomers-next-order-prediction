import requests

url = 'http://localhost:9696/predict'
r = requests.post(url,jsonify({'products': recommended_products, 'Time': difference, 'predict_list':to_predict_list, 'top5':top5_products})

print(r.jsonify())

