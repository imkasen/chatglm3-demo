"""
FastAPI 主文件，创建 API 对象
"""
from fastapi import FastAPI, applications
from fastapi.openapi.docs import get_swagger_ui_html

from .routers import api


# CDN
def swagger_monkey_patch(*args, **kwargs):
    """
    重写方法将 CDN 地址替换为国内网址
    """
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="https://cdn.staticfile.org/swagger-ui/5.6.2/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.staticfile.org/swagger-ui/5.6.2/swagger-ui.css",
    )


applications.get_swagger_ui_html = swagger_monkey_patch

# 创建 FastAPI 对象，并将 swagger 文档从默认 '/docs' 改为 '/'，关闭 redoc 文档
app = FastAPI(title="ChatGLM3-6B FastAPI Demo", description="A ChatGLM3-6B Chat Server", docs_url="/", redoc_url=None)

app.include_router(router=api)
