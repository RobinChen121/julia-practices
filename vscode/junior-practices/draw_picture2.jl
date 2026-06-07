using Plots  # or StatsPlots
using GraphRecipes  # if you wish to use GraphRecipes package too

x = range(0, 10, length=100)
y = sin.(x)
plot(x, y)