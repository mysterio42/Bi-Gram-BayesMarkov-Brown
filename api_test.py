import requests

url_inf = 'http://0.0.0.0:5000/inference'
url_battle = 'http://0.0.0.0:5000/battle'
url_neighs = 'http://0.0.0.0:5000/neighs'

payload_inf = {
    'token': 'last'
}
payload_neighs = {
    'token':'king',
    'top_n':5,
}

if __name__ == '__main__':
    r = requests.post(url_inf, json=payload_inf)
    print(r.text)
    r = requests.get(url_battle)
    print(r.text)
    r = requests.post(url_neighs, json=payload_neighs)
    print(r.text)
