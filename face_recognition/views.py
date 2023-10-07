from django.shortcuts import HttpResponse
from django.http import HttpRequest, JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import numpy as np
import cv2
import time
import os
import shutil
import zipfile
from face_recognition import insightfaceTool

face_tool: insightfaceTool.InsightfaceTool = insightfaceTool.InsightfaceTool()


def upload_image(request: HttpRequest):
    if request.method == 'POST' and request.FILES['image']:
        print("接收到的参数：{}".format(request.POST))
        # 从请求中获取上传的图像文件
        image_file = request.FILES['image']
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({'message': '请上传图片文件！'})
        # 使用OpenCV读取图像文件
        image_data = image_file.read()
        cv2_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite('C:/Users/blackTV/Desktop/{}-{}'.format(time.time(), image_file.name), cv2_image)
        return JsonResponse({'message': '成功'})
    # 如果请求不是POST或者没有上传图像文件，则返回错误响应
    return JsonResponse({'message': '请使用post请求上传图片！'})


# 固定要写一个request入参
def helloWorld(request: HttpRequest):
    print(request.GET)
    return HttpResponse("你好小鬼")


def upload_zip(request: HttpRequest):
    if request.method == 'POST' and request.FILES.get('zip_file'):
        zip_file = request.FILES['zip_file']
        fs = FileSystemStorage()
        filename = fs.save('{}/{}_{}'.format(settings.TEMPORARY_PATH, time.time(), zip_file.name), zip_file)
        uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename)
        # 临时生成一个路径，解压到这里面
        path = '{}/{}'.format(settings.TEMPORARY_PATH, time.time())
        # 解压缩ZIP文件
        with zipfile.ZipFile(uploaded_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # 防止中文乱码
                file_info.filename = file_info.filename.encode('cp437').decode('gbk')
                zip_ref.extract(file_info, path)
        # 将目录下的人脸图片全部解析到索引里
        face_tool.batching(path)
        # 解析完毕后删除临时目录
        shutil.rmtree(path)
        return JsonResponse({'message': '成功 '})
    return JsonResponse({'message': '请使用post请求上传zip压缩包，图片命名格式：【${名称}_%{id}.jpg】，图片格式不局限于jpg'})
