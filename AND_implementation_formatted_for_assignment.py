# AND gate implementation using Hebbian learning
# [x1, x2, ] 
data = [[1, 1, 1, 1],[1, -1, 1, -1],[-1, 1, 1, -1],[-1, -1, 1, -1]] #[]

# initilize weights 
w1=0
w2=0
b=0
for i in range(len(data)):
  
  print("Iteration-"+str(i)+"########")
  print(data[i])
  
  print("w1_new = w1_old + x1 * t ")
  print("       = "+str(w1)+"+"+str(data[i][0])+"*"+str(data[i][3]))
  w1 = w1+data[i][0]*data[i][3]
  print("       = "+str(w1))
  
  
  print("w2_new = w2_old + x2 * t ")
  print("       = "+str(w2)+"+"+str(data[i][1])+"*"+str(data[i][3]))
  w2 = w2+data[i][1]*data[i][3]
  print("       ="+str(w2))
  
  
  print("b_new = b_old + 1 * t ")
  print("      = "+str(b)+"+ 1 * "+str(data[i][3]))
  b = b+1*data[i][3] # bias is 1 for all inputs
  print("      = "+str(b))
