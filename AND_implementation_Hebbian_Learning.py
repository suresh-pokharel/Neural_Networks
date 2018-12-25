# AND gate implementation using Hebbian learning 

data = [[1, 1, 1, 1],[1, -1, 1, -1],[-1, 1, 1, -1],[-1, -1, 1, -1]] #[]

w1=0
w2=0
b=0
for i in range(4):
  w1 = w1+data[i][0]*data[i][3]
  w2 = w2+data[i][1]*data[i][3]
  b = b+1*data[i][3] # bias is 1 for all inputs

  print("Iteration-"+str(i)+"########")
  print(data[i])
  print("w1="+str(w1))
  print("w2="+str(w2))
  print("b="+str(b))

  
  # OUTPUT
# Iteration-0########
# [1, 1, 1, 1]
# w1=1
# w2=1
# b=1
# Iteration-1########
# [1, -1, 1, -1]
# w1=0
# w2=2
# b=0
# Iteration-2########
# [-1, 1, 1, -1]
# w1=1
# w2=1
# b=-1
# Iteration-3########
# [-1, -1, 1, -1]
# w1=2
# w2=2
# b=-2
