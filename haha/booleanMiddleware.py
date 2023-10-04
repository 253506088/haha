from django.http import QueryDict
from django.utils.deprecation import MiddlewareMixin

"""
格式化POST和GET请求的Boolean类型的入参
"""


class ConvertBooleanMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if request.method == 'POST':
            # post请求，这里处理post表单请求，json入参的没有这个问题不用处理。
            mutable_post = QueryDict('', mutable=True)
            mutable_post.update(request.POST)
            for key, value in mutable_post.items():
                if value.lower() == 'true':
                    mutable_post[key] = True
                elif value.lower() == 'false':
                    mutable_post[key] = False
            request.POST = mutable_post
        if request.method == 'GET':
            # get请求
            mutable_get = QueryDict('', mutable=True)
            mutable_get.update(request.GET)
            for key, value in mutable_get.items():
                if value.lower() == 'true':
                    mutable_get[key] = True
                elif value.lower() == 'false':
                    mutable_get[key] = False
            request.GET = mutable_get
