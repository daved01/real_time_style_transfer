import csv
import os.path
from datetime import datetime
import matplotlib.pyplot as plt


class LossLogger:
    def __init__(self, path, epochs_ran=0):
        # Inner epoch accumulated loss
        self._step_loss = 0
        self._step_counter = 0

        # Averages
        self._average_losses = []
        self._timestamps = []
        self._epochs = []

        # Check if file exists. If so read it, else create it.
        self._path = path
        self._filename = path.split("/")[-1]
        self._fileextension = ".log"
        self._epochs_ran = epochs_ran
        
        # Check if log file exists.
        read_file = os.path.exists(self._path + "/" + self._filename + self._fileextension)
        if read_file == True:
            # Read file
            with open(self._path + "/" + self._filename + self._fileextension, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(",")
                    if len(line) == 3:
                        try:
                            self._epochs.append(int(line[0].strip("\n")))
                            self._average_losses.append(float(line[1].strip("\n")))
                            self._timestamps.append(str(line[2]).strip("\n"))
                        except ValueError:
                            pass
    
            # Check lengths.
            assert(len(self._epochs) == len(self._average_losses))
            assert(len(self._epochs) == len(self._timestamps))

            # Check the epochs.
            try:
                if self._epochs[-1] != epochs_ran:
                    print("Error! Last saved epoch is {:.0f}, but given weights are from epoch {:.0f}! Please start at epoch {:.0f}.".format(
                        self._epochs[-1], epochs_ran, epochs_ran)
                        )
                    exit()
            except IndexError:
                pass


    def add(self, loss):
        """
        Adds loss to accumulated loss.
        """
        
        self._step_loss += loss
        self._step_counter += 1

    
    def log_average_loss(self):
        """
        Calculates average loss for epoch and resets the step couter.
        """
        
        if self._step_counter == 0:
            return
        
        # Get current date and time.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Calculate average loss
        avg_loss = self._step_loss / self._step_counter

        # Append average loss to other averages
        if len(self._epochs) == 0:
            self._epochs.append(self._epochs_ran + 1)
        else:
            self._epochs.append(self._epochs[-1] + 1)
        self._average_losses.append(avg_loss.numpy())
        self._timestamps.append(timestamp)

        # Reset steps counters
        self._step_loss = 0
        self._step_counter = 0


    def save(self):
        """
        Calculates latest average epoch loss and writes all averages to disk.
        """
        
        self.log_average_loss()

        # Format: Epoch, loss, date
        assert len(self._average_losses) == len(self._timestamps), "Number of average losses and timestamps must be equal!"
        assert len(self._epochs) == len(self._timestamps), "Number of epochs and timestamps must be equal!"

        # Save to file.
        with open(self._path + "/" + self._filename + self._fileextension, 'w') as f:
            # Write header.
            f.writelines("Pretraining weights: \n")
            f.writelines(["Epochs, ", "Loss, ", "Timestamp"])
            for i in range(len(self._epochs)):
                f.writelines(["\n", str(self._epochs[i]), ",",str(self._average_losses[i]), ",", self._timestamps[i]])


    def plot(self):
        """
        Plots the loss over the epochs and saves the plot in the folder.

        Name: 
        loss_<model_name>_<dataset_name>_<style_image>_batchsize<batch_size>_epochs<num_epochs>.png
        """

        fig = plt.figure()
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(self._epochs, self._average_losses, "o-", color='blue')
        fig.savefig(self._path + "/" + self._filename + ".png", transparent=False)