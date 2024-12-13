import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 20
import numpy as np

# SGD log
sgd_loss = []
sgd_grad_norm = []
sgd_steps = []
with open('../logs/MNLI-roberta-large-prompt-standard-k512-roberta-large-ft-sgdseed42-bs64-lr1e-4-step10000-evalstep1000-512-42.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        line = line.strip()
        if "loss" in line and "norm" in line:
            item = line.split("- src.trainer -")[-1].strip()
            item = eval(item)
            sgd_loss.append(item['loss'])
            sgd_grad_norm.append(item['norm'])
            sgd_steps.append(count * 10)
            count += 1

print(len(sgd_loss))
print(sgd_loss)
print(sgd_steps)
print(sgd_grad_norm)

# MeZO log
zo_loss = []
zo_grad_norm = []
zo_steps = []
with open('../logs/MNLI-roberta-large-prompt-standard-k512-roberta-large-mezo-ft-testseed42-bs64-lr1e-6-eps1e-3-wd0-step100000-evalstep10000-512-42.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if "loss" in line and "grad_norm" in line:
            item = line.split("- src.trainer -")[-1].strip()
            item = eval(item)
            zo_loss.append(item['loss'])
            zo_grad_norm.append(item['grad_norm'])
            zo_steps.append(item['global_step'])

print(len(zo_loss))
print(zo_loss)
print(zo_steps)
print(zo_grad_norm)

sgd_x = []
sgd_y = []
for i, step in enumerate(sgd_steps):
    if step % 100 == 0:
        sgd_x.append(step)
        sgd_y.append(sgd_grad_norm[i])

zo_x = []
zo_y = []
for i, step in enumerate(zo_steps):
    if step % 100 == 0:
        zo_x.append(step)
        zo_y.append(zo_grad_norm[i])
        # if step >= 10000:
        #     break

plt.plot(sgd_x, sgd_y, lw=1.5)
plt.xlabel("Steps")
plt.ylabel("Gradient Norm")
plt.subplots_adjust(bottom=0.15, left=0.15)
# plt.title('SGD')
plt.savefig('sgd_grad_norm.pdf')
plt.show()

print(max(zo_y))
plt.plot(zo_x, zo_y, lw=1.5, color='#C45B45')
plt.xlabel("Steps")
plt.ylabel("Gradient Norm")
plt.subplots_adjust(bottom=0.15, left=0.16)
# plt.yticks(list(range(0, 1200000, 250000)))
# plt.title('MeZO')
plt.savefig('zo_grad_norm.pdf')
plt.show()


