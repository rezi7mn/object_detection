import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# 認証情報の取得
try:
    KEY = st.secret['KEY']
    ENDPOINT = st.secrets['ENDPOINT']
except KeyError:
    print("Missing environment variable 'ENDPOINT' or 'KEY'")
    print("Set them before running this sample.")
    exit()

# クライアントの初期化（新しいSDKのクラス）
client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

st.title('物体検出アプリ (Updated)')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg','png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # 画像データをバイナリとして読み込み
    image_data = uploaded_file.getvalue()

    # API呼び出し（物体検出とタグ付けを一度に実行）
    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.TAGS],
    )

    # 描画準備
    draw = ImageDraw.Draw(img)
    # フォント設定（パスやサイズは適宜調整してください）
    try:
        font = ImageFont.truetype(font='./Helvetica 400.ttf', size=32)
    except:
        font = ImageFont.load_default()

    # 1. 物体検出結果の処理
    if result.objects is not None:
        for obj in result.objects.list:
            # 新しいSDKでは座標が r.x, r.y, r.w, r.h に格納されています
            r = obj.bounding_box
            caption = obj.tags[0].name # 最も信頼度の高いラベル名
            
            # テキストの背景矩形サイズ計算
            bbox = draw.textbbox((r.x, r.y), caption, font=font)
            
            # 枠線とラベルの描画
            draw.rectangle([(r.x, r.y), (r.x + r.width, r.y + r.height)], outline='green', width=5)
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], fill='green')
            draw.text((r.x, r.y), caption, fill='white', font=font)

    # メイン画像の表示
    st.image(img)

    # 2. タグ情報の表示
    if result.tags is not None:
        tags_name = [tag.name for tag in result.tags.list]
        tags_str = ', '.join(tags_name)
        st.markdown('**認識されたコンテンツタグ**')
        st.markdown(f'> {tags_str}')  