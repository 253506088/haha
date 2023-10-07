from django.db import models
from django.utils import timezone


# 用户类
class User(models.Model):
    # id这一列不写的话Django也会帮你自动带上
    id = models.AutoField(primary_key=True, help_text="自增id")
    name = models.CharField(max_length=64, help_text="名称")
    face_vector = models.BinaryField(help_text="512维度面部向量二维化数据")
    createTime = models.DateTimeField(default=timezone.now(), help_text="创建时间")
    updateTime = models.DateTimeField(default=timezone.now(), help_text="变更时间")
