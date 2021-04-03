from graphviz import render
import glob
for each_file in glob.glob("*.dot"):
    print(each_file)
    render('dot', 'png', each_file)  
