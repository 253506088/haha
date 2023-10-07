import os
import re
import cv2
import insightface
import time
import numpy as np
from sklearn import preprocessing
import faiss
from django.conf import settings

"""
人脸对比工具类
"""


class InsightfaceTool:
    # insightface检测模型
    model = None
    # faiss索引
    faiss_index = None
    # 人脸数组
    face_array = None
    # id数组
    id_array = None

    def __init__(self):
        # 加载默认的人脸识别模型，当 allowed_modules=['detection', 'recognition'] 时，只单纯检测和识别，没有年龄性别等信息，会快一点
        self.model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
        # 使用GPU进行运算，需要安装好这三个：[cuda、cuDNN、(TensorFlow 或 PyTorch，我这里是PyTorch)]
        self.model.prepare(ctx_id=settings.GPU_NUMBER, det_thresh=0.5, det_size=(640, 640))
        if os.path.exists(settings.FAISS_INDEX_SAVE_PATH):
            print("文件存在，读取落盘索引")
            self.faiss_index = faiss.read_index(settings.FAISS_INDEX_SAVE_PATH)
            load_data = np.load(settings.NDARRAY_SAVE_PATH)
            self.face_array = load_data['face_array']
            self.id_array = load_data['id_array']
            print("666")
        else:
            # 创建向量索引
            self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(512))

    def batching(self, face_image_path):
        """
        批量处理
        """
        start_time = time.time()
        for file_name in os.listdir(face_image_path):
            # 检查文件是否为图像文件
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                file_path = "{}/{}".format(face_image_path, file_name)
                # 使用二进制模式打开文件
                with open(file_path, 'rb') as f:
                    # 读取图像数据
                    image_data = f.read()
                # 使用cv2.imdecode解码图像数据
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite("{}/{}".format('./6666', file_name), image)
                idAndName = self.getIdAndName(file_name)
                self.extractVectorSingleFaceAndSave(image, idAndName['id'], False)
        self.diskStorage()
        print("批量处理结束，用时 {} 秒".format(time.time() - start_time))

    def extractVectorSingleFaceAndSave(self, image, id, save_index=True):
        """
        提取只包含单张人脸图片中的人脸向量数据，当【save_index=True】则向量写入到索引中
        """
        face = self.extractVectorSingleFace(image)
        if face is None:
            return
        id_array = np.array([id]).astype('i8')
        self.faiss_index.add_with_ids(face, id_array)
        if self.face_array is None:
            self.face_array = np.array(face)
            self.id_array = np.array(id_array)
        else:
            self.face_array = np.append(self.face_array, face, axis=0)
            self.id_array = np.append(self.id_array, id_array)
        if save_index:
            self.diskStorage()

    def extractVectorSingleFace(self, image):
        """
        提取只包含单张人脸图片中的人脸向量数据
        """
        # 人脸向量解析
        faces = self.model.get(image)
        if len(faces) != 1:
            print("图片里未检测到人脸")
            return None
        # 归一化
        face = self.normalize(faces[0].embedding)
        print("\nface数组形状：{}，dtype：{}".format(face.shape, face.dtype))
        return face

    def normalize(self, embedding):
        """
        人脸向量归一化处理
        """
        # 原数组shape为(512,)，而归一化处理接受二维数组，所以转换为二维数组，返回数组形状为(1,512)
        face = np.array(embedding).reshape((1, -1))
        # 进行归一化，将特征向量的值处理到0~1之间
        face2 = preprocessing.normalize(face)
        return face2

    def diskStorage(self):
        """
        落盘保存
        """
        faiss.write_index(self.faiss_index, settings.FAISS_INDEX_SAVE_PATH)
        np.savez(settings.NDARRAY_SAVE_PATH, face_array=self.face_array, id_array=self.id_array)

    def faceComparison(self, unknown_face):
        """
        单张人脸对比，匹配成功返回对应的id，匹配失败返回None
        """
        distances, indices = self.faiss_index.search(unknown_face, k=1)
        id = indices[0][0]
        distance = distances[0][0]
        distance_format = f"{distance:.8f}"  # 保留8位小数，避免float类型科学计数法让你看不懂
        if distance <= settings.FAISS_THRESHOLD:
            print("\n匹配成功，匹配到id:{} ; distance:{}".format(id, distance_format))
            return id
        else:
            print("\n匹配失败，匹配到id:{} ; distance:{}".format(id, distance_format))
            return None

    def getIdAndName(self, file_name):
        # 定义正则表达式模式，提取规则：${名称}_%{id}.${后缀名}
        pattern = r'(.+?)_(\d+)\.(.+)'
        # 使用正则表达式进行匹配和提取
        matches = re.findall(pattern, file_name)
        # 提取结果
        if matches:
            name = matches[0][0]
            id = matches[0][1]
            suffix = matches[0][2]
            print("名称:", name)
            print("ID:", id)
            print("后缀名:", suffix)
            return {
                "id": id,
                "name": name,
                "suffix": suffix,
            }
        else:
            print("未找到匹配的内容。")
            return None
