latex_text = r'''
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\linewidth]{png/chapter3/ShallowReLU.png}
    \caption{Figure 3.1}
\end{figure}

图 3.1 整流线性单元 (Rectified Linear Unit, ReLU)。这种激活函数在输入小于零时输出为零，否则保持输入值不变。简而言之，它将所有负数输入值变为零。需要注意的是，虽然有许多其他激活函数可供选择（参见图 3.13），但 ReLU 由于其简单易懂，成为最常用的选择。

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\linewidth]{png/chapter3/ShallowFunctions.png}
    \caption{Figure 3.2}
\end{figure}

图 3.2 由方程 3.1 定义的函数族。a-c) 展示了三种不同参数 \(\phi\) 的选择下的函数。在这些函数中，输入与输出的关系均为分段线性。不过，各个拐点的位置、拐点间线段的斜率，以及整体高度各不相同。

'''

def replace_captions(text):
    lines = text.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith('\\begin{figure}'):
            figure_block = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith('\\end{figure}'):
                figure_block.append(lines[i])
                i += 1
            if i < len(lines):
                figure_block.append(lines[i])  # Append the \end{figure} line
            i += 1

            # Find the caption text
            caption_text = ""
            if i < len(lines) and lines[i].startswith('图'):
                caption_text = lines[i].strip()
                i += 1

            if caption_text:
                figure_block = [line if not line.startswith('\\caption{') else f'\\caption{{{caption_text}}}' for line in figure_block]

            result.extend(figure_block)
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)

updated_latex_text = replace_captions(latex_text)
print(updated_latex_text)
