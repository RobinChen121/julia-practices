using Plots

# 交互式绘图，会在浏览器中显示，自动支持中文
# run in new process 才会在浏览器中弹出图形，否则 julia 绘图面板弹出
plotly()

# 生成数据
x = 0:0.1:10
y = sin.(x)

# 绘图
p = plot(x, y, title="正弦函数", label="sin(x)", lw=3, xlabel="x", ylabel="y")

# 显示图像，不是很有必要
display(p)
 
# # 下面代码避免在非 REPL（交互式环境） 中运行时一闪而过
# println("按下回车键以关闭图像并结束程序...")
# readline() # 程序会在此处暂停，等待输入