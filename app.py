from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

import json

with open('secret.json') as f:
    secret = json.load(f)

KEY = secret['KEY']
ENDPOINT = secret['ENDPOINT']

# key,endpoint渡して認証
computervision_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(KEY))

def get_tags(filepath):
    local_image = open(filepath, "rb")

    tags_result = computervision_client.tag_image_in_stream(local_image)
    tags = tags_result.tags
    tags_name = []
    for tag in tags:
        tags_name.append(tag.name)
        
    return tags_name

def detect_objects(filepath):
    local_image = open(filepath, "rb")

    detect_objects_results = computervision_client.detect_objects_in_stream(local_image)
    objects = detect_objects_results.objects
    return objects

import streamlit as st
from PIL import ImageDraw
from PIL import ImageFont

st.title('物体検出アプリ')

# file_uploaderではファイルのパスは取得できないので画像をフォルダに保存して取得
uploaded_file = st.file_uploader('Choose an image...', type=['jpg','png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_path = f'img/{uploaded_file.name}'
    img.save(img_path)
    objects = detect_objects(img_path)

    # 描画
    draw = ImageDraw.Draw(img)
    for object in objects:
        x = object.rectangle.x
        y = object.rectangle.y
        w = object.rectangle.w
        h = object.rectangle.h
        # 文字が入ってる
        caption = object.object_property

        # 矩形左上のテキスト, フォントファイルをttfで保存
        font = ImageFont.truetype(font='./Helvetica 400.ttf', size=32)
        # テキストサイズ分からないと背景の矩形が作れない
        # (左上の点のx座標, 左上の点のy座標, 右下の点のx座標, 右下の点のy座標)を返す。
        bbox = draw.textbbox((x, y), caption, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        draw.rectangle([(x, y), (x+w, y+h)], fill=None, outline='green', width=5)
        draw.rectangle([(x, y), (x+width, y+height)], fill='green')
        draw.text((x, y), caption, fill='white', font=font)

    st.image(img)

    tags_name = get_tags(img_path)
    tags_name = ', '.join(tags_name)
    st.markdown('**認識されたコンテンツタグ**')
    st.markdown(f'> {tags_name}')    