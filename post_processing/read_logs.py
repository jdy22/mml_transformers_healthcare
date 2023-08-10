from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

logfile = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/Code/individual_project/runs_modality/rerun2-3/events.out.tfevents.1690282310.gpu01.doc.ic.ac.uk"

event_acc = EventAccumulator(logfile)
event_acc.Reload()

# Training loss
train_epochs = []
train_loss = []
for item in event_acc.Scalars("train_loss"):
    train_epochs.append(item.step) 
    train_loss.append(item.value)

# Validation accuracy
val_epochs = []
val_acc = []
for item in event_acc.Scalars("val_acc"):
    val_epochs.append(item.step)
    val_acc.append(item.value)

max_acc = max(val_acc)
print(f"Maximum validation accuracy = {max_acc} at epoch number {val_epochs[val_acc.index(max_acc)]+1}")

# Plot
fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(8, 3.5))

ax1.plot(train_epochs, train_loss, "r")
ax1.set_title("Training loss")
ax1.set_xlabel("Epochs")

ax2.plot(val_epochs, val_acc, "c")
ax2.set_title("Validation accuracy")
ax2.set_xlabel("Epochs")

plt.show()