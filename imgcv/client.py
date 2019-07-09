# coding:utf-8

import os
import json,requests
from urllib import request

def get_token_baiduAi(ak,sk):
    host= 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+ ak+'&client_secret='+sk
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    req = request.Request(host)
    req.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = request.urlopen(req)
    content = response.read()
    content.decode('utf-8')
    jsonData = json.loads(content)
    if (content):
        return jsonData['access_token']
    else:
        raise Exception("Token get filed",jsonData)

def PostWithForm(Host, Param, Token):
    '''
    post with application/x-www-form-urlencoded
    '''
    Host = Host + '?access_token=' + Token
    req = request.Request(Host, Param)
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    res = requests.post(Host,Param)
    return res