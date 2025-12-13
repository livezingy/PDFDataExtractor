# 部署指南

本文档说明如何部署PDF Table Extractor应用。

## 快速选择

- **一键试用（推荐）**：使用 [Streamlit Cloud部署](streamlit_cloud_deployment.md) - 支持PDFPlumber、Camelot、PaddleOCR
- **完整功能**：本地部署 - 支持所有功能包括Transformer
- **生产环境**：服务器部署 - 高性能、可扩展

## 部署方式

### 1. Streamlit Cloud部署（一键试用）⭐

**推荐用于快速试用和演示**

- ✅ 免费
- ✅ 一键部署
- ✅ 支持PDFPlumber、Camelot、PaddleOCR
- ❌ 不支持Transformer（资源限制）

详细步骤请参考 [Streamlit Cloud部署指南](streamlit_cloud_deployment.md)

### 2. 本地部署（开发环境）

#### 1.1 基本部署

```bash
# 1. 克隆项目
git clone https://github.com/livezingy/PDFDataExtractor.git
cd PDFDataExtractor

# 2. 创建虚拟环境（推荐）
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
pip install -r requirements_streamlit.txt

# 5. 启动应用
streamlit run streamlit_app/streamlit_app.py
```

#### 1.2 配置选项

创建 `.streamlit/config.toml` 文件（如果不存在）：

```toml
[server]
port = 8501
address = "0.0.0.0"  # 允许外部访问
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### 2. 服务器部署

#### 2.1 使用systemd（Linux）

创建服务文件 `/etc/systemd/system/pdf-extractor.service`：

```ini
[Unit]
Description=PDF Table Extractor Streamlit App
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/PDFDataExtractor
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run streamlit_app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable pdf-extractor
sudo systemctl start pdf-extractor
```

#### 2.2 使用Nginx反向代理

Nginx配置示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

### 3. Docker部署

#### 3.1 创建Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt requirements_streamlit.txt ./

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_streamlit.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "streamlit_app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 3.2 构建和运行

```bash
# 构建镜像
docker build -t pdf-extractor .

# 运行容器
docker run -d \
  -p 8501:8501 \
  --name pdf-extractor \
  -v /path/to/models:/app/models \
  pdf-extractor
```

#### 3.3 Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  pdf-extractor:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

运行：

```bash
docker-compose up -d
```

### 4. Streamlit Cloud部署

#### 4.1 准备仓库

1. 确保代码已推送到GitHub
2. 确保 `requirements.txt` 和 `requirements_streamlit.txt` 存在

#### 4.2 部署步骤

1. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
2. 连接GitHub账户
3. 选择仓库
4. 配置部署：
   - Main file path: `streamlit_app/streamlit_app.py`
   - Python version: 3.10
5. 点击"Deploy"

**注意**：Streamlit Cloud有资源限制，大型模型可能无法运行。

## 环境配置

### 环境变量

可以设置以下环境变量：

```bash
# 模型路径
export MODEL_DIR=/path/to/models

# GPU设置
export CUDA_VISIBLE_DEVICES=0

# 日志级别
export LOG_LEVEL=INFO

# 文件大小限制（MB）
export MAX_FILE_SIZE_MB=10
```

### 配置文件

创建 `config/config.json`（参考 `config/config.example.json`）：

```json
{
  "max_file_size_mb": 10,
  "default_method": "pdfplumber",
  "default_engine": "paddleocr",
  "use_gpu": false,
  "log_level": "INFO"
}
```

## 性能优化

### 1. 使用GPU加速

**PaddleOCR**：
```python
# 在代码中设置
engine = EngineFactory.create_detection('paddleocr', use_gpu=True)
```

**Transformer**：
```bash
# 确保CUDA正确安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 模型缓存

首次使用会下载模型，后续会使用缓存：

```bash
# 模型缓存位置
# Linux/Mac: ~/.cache/
# Windows: C:\Users\<username>\.cache\
```

### 3. 内存优化

- 处理较小的文件
- 一次处理一页
- 关闭不需要的引擎
- 使用CPU版本（如果不需要GPU）

### 4. 并发处理

对于多用户场景，考虑：

- 使用负载均衡（Nginx）
- 使用多个Streamlit实例
- 使用异步处理（计划中）

## 安全注意事项

### 1. 文件上传限制

- 设置文件大小限制
- 验证文件类型
- 扫描上传文件（如果可能）

### 2. 访问控制

- 使用HTTPS
- 配置身份验证（如果需要）
- 限制访问IP（如果需要）

### 3. 资源限制

- 设置处理超时
- 限制并发请求
- 监控资源使用

### 4. 数据隐私

- 不存储用户上传的文件
- 处理完成后删除临时文件
- 遵守数据保护法规

## 监控和维护

### 1. 日志管理

配置日志：

```python
# 在代码中配置
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### 2. 健康检查

创建健康检查端点（计划中）：

```python
# 示例
@app.route('/health')
def health():
    return {'status': 'healthy'}
```

### 3. 性能监控

- 监控CPU和内存使用
- 监控处理时间
- 监控错误率

### 4. 备份

- 定期备份配置
- 备份模型文件（如果自定义）
- 备份日志文件

## 故障排除

### 常见问题

1. **端口被占用**：
```bash
# 查找占用端口的进程
lsof -i :8501  # Linux/Mac
netstat -ano | findstr :8501  # Windows

# 使用其他端口
streamlit run streamlit_app/streamlit_app.py --server.port 8502
```

2. **内存不足**：
- 减少并发处理
- 使用较小的文件
- 增加服务器内存

3. **模型加载失败**：
- 检查网络连接
- 手动下载模型
- 检查磁盘空间

## 更新和维护

### 更新应用

```bash
# 拉取最新代码
git pull origin main

# 更新依赖
pip install --upgrade -r requirements.txt

# 重启服务
sudo systemctl restart pdf-extractor  # systemd
# 或
docker-compose restart  # Docker
```

### 清理

```bash
# 清理Python缓存
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# 清理模型缓存（如果需要）
rm -rf ~/.cache/huggingface/
```

---

**提示**：部署前请确保阅读 [用户使用指南](streamlit_user_guide.md) 和 [FAQ](FAQ.md)。
