"""
以 API 服务的形式运行 LLM
"""
from api import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        # app="api.main:app",
        app=app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
