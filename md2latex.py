import re

def convert_markdown_to_latex(md_text):
    # 转换一级标题，去掉序号
    latex_text = re.sub(r'^# \d+(\.\d+)*\s*(.+)', r'\\chapter{\2}', md_text, flags=re.MULTILINE)
    
    # 转换二级标题，去掉序号
    latex_text = re.sub(r'^## \d+(\.\d+)*\s*(.+)', r'\\section{\2}', latex_text, flags=re.MULTILINE)
    
    # 转换三级标题，去掉序号
    latex_text = re.sub(r'^### \d+(\.\d+)*\s*(.+)', r'\\subsection{\2}', latex_text, flags=re.MULTILINE)
    
    # 转换行内公式
    latex_text = re.sub(r'(?<!\\)\$(.+?)\$', r'\\(\1\\)', latex_text)
    
    # 转换单行公式，保留align环境
    def replace_math(match):
        content = match.group(1)
        if '\\begin{align}' in content or '\\end{align}' in content:
            return f"{content}"
        else:
            return f"\\[{content}\\]"
    
    latex_text = re.sub(r'(?<!\\)\$\$(.+?)\$\$', replace_math, latex_text, flags=re.DOTALL)
    
    # 转换加粗文本
    latex_text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', latex_text)
    
    # 转换图片，并将 figures 替换为 png，将 svg 后缀替换为 png
    latex_text = re.sub(r'!\[([^\]]*)\]\(figures/([^)]+)\.svg\)', r'\\begin{figure}[h!]\n\\centering\n\\includegraphics[width=0.7\\linewidth]{png/\2.png}\n\\caption{\1}\n\\end{figure}', latex_text)
    
    return latex_text

# 示例Markdown文本
with open('Chapter 3 Shallow neural networks.md', 'r') as m:
    md_text = m.read()

latex_text = convert_markdown_to_latex(md_text)

# 输出转换后的LaTeX文本
print(latex_text)

# # 将转换后的内容保存到LaTeX文件latex/Tex_files/
with open("chapter03.tex", "w") as f:
    f.write(latex_text)
