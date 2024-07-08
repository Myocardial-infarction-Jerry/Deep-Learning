import urllib.parse
import requests


# LaTeX 公式
fileName = "ResNet.svg"
latex_formula = r'''
\begin{align}
H(x)=F(x)+x
\end{align}
'''


# URL 编码 LaTeX 公式
encoded_formula = urllib.parse.quote(latex_formula)

# 构造请求 URL
url = f"https://latex.codecogs.com/svg.latex?{encoded_formula}"

# 发送 HTTP 请求并下载 SVG
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 保存为 SVG 文件
    with open(fileName, "wb") as file:
        file.write(response.content)
    print(f"SVG file saved as {fileName}")
else:
    print(f"Failed to retrieve SVG. HTTP status code: {response.status_code}")
