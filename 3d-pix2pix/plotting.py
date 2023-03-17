
import matplotlib.pyplot as plt
  
x = []
y = []
for line in open('train_log.txt', 'r'):
    lines = [i for i in line.split()]
    x.append(lines[1])
    y.append(int(lines[3]))
      
plt.title("Generator loss")
plt.xlabel('Epoch')
plt.ylabel('G_loss')
plt.yticks(y)
plt.plot(x, y, marker = 'o', c = 'g')
  
plt.savefig('genloss.png')

x = []
y = []
for line in open('train_log.txt', 'r'):
    lines = [i for i in line.split()]
    x.append(lines[1])
    y.append(int(lines[5]))
      
plt.title("Discriminator loss")
plt.xlabel('Epoch')
plt.ylabel('D_loss')
plt.yticks(y)
plt.plot(x, y, marker = 'o', c = 'b')
  
plt.savefig('disloss.png')
