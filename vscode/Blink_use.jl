using Interact: slider, on, vbox, @dom_str
using PlotlyJS: plot, restyle!, scatter
using Blink: Window, body!

# PlotlyJS 中的画图语法与 Plot.jl 中的不太一样

win = Window() # 打开一个窗口
x = collect(0.0:0.01:2π) # collect 将一个等差数列转化为数组
y = sin.(x)

trace = scatter(x=x, y=y)
p = plot(trace)

function updateplot(i)
    y = sin.(i*x)
    restyle!(p, 1, y=[y])
end

sli = slider(1:100, label="i", value = 10)
on(updateplot, sli)

ui = dom"div"(vbox(sli, p))
body!(win, ui) # 将内容生成到窗口里