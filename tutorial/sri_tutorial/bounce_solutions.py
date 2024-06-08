


# input_dim, output_dim = 4, 4
# print("Data has been loaded.")

# print("Now, you have to add the code to actually search for a program using NEAR")
# print(input_dim, output_dim)


print("Defined NEAR")




# And then the code below plots it to show how it compares to a trajectory in the training set.
import matplotlib.pyplot as plt

title = "Bouncing ball (ground truth in black)"

plt.figure(figsize=(8, 8))

print(trajectory[:])
plt.scatter(trajectory[:, 0], trajectory[:, 1], marker="o", color="C0")

plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.2, color="C0")

plt.scatter(trajectoryb[:, 0], trajectoryb[:, 1], marker="o", color="C1")

plt.plot(trajectoryb[:, 0], trajectoryb[:, 1], alpha=0.2, color="C1")

truth = datamodule.train.inputs[0, :, :]

print(truth[0, :])

plt.scatter(truth[:, 0], truth[:, 1], marker="o", color="black")

plt.plot(truth[:, 0], truth[:, 1], alpha=0.2, color="black")


plt.title(title)
plt.xlim(-5, 10)
plt.ylim(-5, 7)
plt.grid(True)
plt.show()
