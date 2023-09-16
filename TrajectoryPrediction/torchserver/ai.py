import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from control_params import params as cp
import datetime

torch.manual_seed(cp['RANDOM_SEED'])
np.random.seed(cp['RANDOM_SEED'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cp['DIM_INPUT'], 
            hidden_size=cp['EMBEDDING_SIZE'], 
            batch_first=True
        )
        self.dropout = nn.Dropout(cp['DROPOUT'])
        self.fc = nn.Linear(
            cp['EMBEDDING_SIZE'], cp['FUTURE_TARGET'] * cp['NUM_OUTPUTS']
        )
        self.activation = nn.LeakyReLU()
        # self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.zero_()
    
    def forward(self, x):
        self.lstm.flatten_parameters()
        out_seq, (h_n, c_n) = self.lstm(x)
        x = torch.squeeze(h_n)
        # print(f"{x.shape}\n{x}")
        x = self.dropout(x)
        # print(x)
        x = self.fc(x)
        # x = self.activation(x)
        x = x.reshape(-1, cp['FUTURE_TARGET'], cp['NUM_OUTPUTS'])
        return x


class AI:
    def __init__(self, robot_id, filename) -> None:
        self.robot_id = robot_id
        self.filename = filename
        self.epoch = 0
        self.device = device
        self.model = Model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=cp['LEARNING_RATE']
        )
    
    def recreate_model(self):
        self.model = Model().to(self.device)
    
    def create_series_examples_from_batch(
            self, dataset, start_index, end_index, past_history, future_target
        ):
        data = []
        labels = []
        list_dataset = list(dataset.values())
        array_dataset = np.asarray(list_dataset)
        for i in range(start_index, end_index):
            data.append(array_dataset[i][:past_history])
            labels.append(array_dataset[i][past_history:past_history+future_target])
        data = np.asarray(data).reshape(end_index-start_index, past_history, 2)
        labels = np.asarray(labels).reshape(
            end_index-start_index, future_target , 2
        )
        
        return data, labels

    def create_training_and_val_batch(
            self, batch, past_history=cp['PAST_HISTORY'], 
            future_target=cp['FUTURE_TARGET'], 
            input_dimension=cp['DIM_INPUT'], 
            train_ratio = cp['TRAIN_RATIO']
    ):
        x_train = np.zeros((1, past_history, input_dimension))
        y_train = np.zeros((1, future_target, input_dimension))
        x_val = np.zeros((1, past_history, input_dimension))
        y_val = np.zeros((1, future_target, input_dimension))

        for v in batch.values():
            tot_samples = len(v)
            print("total number of samples:", tot_samples)
            train_split = round(train_ratio * tot_samples)
            print("training samples:", train_split)
            x_train_tmp, y_train_tmp = self.create_series_examples_from_batch(
                v, 0, train_split, past_history, future_target
            )
            x_val_tmp, y_val_tmp = self.create_series_examples_from_batch(
                v, train_split, tot_samples, past_history,future_target
            )
            x_train = np.concatenate([x_train, x_train_tmp], axis=0)
            y_train = np.concatenate([y_train, y_train_tmp], axis=0)
            x_val = np.concatenate([x_val, x_val_tmp], axis=0)
            y_val = np.concatenate([y_val, y_val_tmp], axis=0)

        return x_train[1:,:,:], x_val[1:,:,:], y_train[1:,:,:], y_val[1:,:,:]

    def train_model(self, train_set: DataLoader, steps: int):
        step = 0
        epoch_loss = 0
        self.epoch += 1
        
        if steps == 0:
            return [1] * 2912, 0, self.epoch, 0
        
        self.model.train()
        start = datetime.datetime.now()
        while step < steps:
            for batch_inputs, batch_labels in train_set:
                output = self.model(batch_inputs)
                loss = self.criterion(output, batch_labels)
                loss.backward()
                epoch_loss += loss.item()
                step += 1
        # self.optimizer.step()
        delta = float((datetime.datetime.now() - start).total_seconds())

        gradients = self.params_to_list(samples=steps, option='grad')
        self.optimizer.zero_grad()
        
        return gradients, epoch_loss / step, self.epoch, delta
    
    def val_model(self, val_set: DataLoader):
        if len(val_set) == 0:
            return 0
        epoch_loss = 0
        self.model.eval()
        for batch_inputs, batch_labels in val_set:
            output = self.model(batch_inputs)
            loss = self.criterion(output, batch_labels)
            epoch_loss += loss.item()
        return epoch_loss / len(val_set)

    def create_dataset(self, x_train, x_val, y_train, y_val):
        x_train = torch.from_numpy(x_train).to(self.device, dtype=torch.float)
        y_train = torch.from_numpy(y_train).to(self.device, dtype=torch.float)
        
        x_val = torch.from_numpy(x_val).to(self.device, dtype=torch.float)
        y_val = torch.from_numpy(y_val).to(self.device, dtype=torch.float)
        
        train_set = TensorDataset(x_train, y_train)
        train_set_loader = DataLoader(train_set, batch_size=cp['LOCAL_BATCH'])

        val_set = TensorDataset(x_val, y_val)
        val_set_loader = DataLoader(val_set)

        return train_set_loader, val_set_loader

    def load_data(self, filenames):
        """filter data v3

        Returns:
            _type_: _description_
        """
        count = 0
        samples = {count: []}
        for filename in filenames:
            with open(filename, 'r') as f:
                next(f)
                for line in f:
                    data = line.split(',')
                    if len(data) == 5:
                        if not samples[count]:
                            previous_id = float(data[1])
                            previous_time = float(data[2]) - 1
                        else:
                            previous_id = current_id
                            previous_time = current_time

                        current_id = float(data[1])
                        current_time = float(data[2])

                        x1 = float(data[3])
                        x2 = float(data[4])
                        if current_time - previous_time == 1 and previous_id == current_id:
                            samples[count].append((x1, x2))
                            if len(samples[count]) == 100:
                                count+=1
                                samples[count] = []
                        else: 
                            samples[count] = [(x1, x2)]
                    else:
                        samples[count] = [] # if there is an issue (file that is being written at the same time it is read, errors may occure)

                if len(samples[len(samples.keys())-1]) != 100:
                    # delete last empty trajectory
                    samples[count] = []

        if len(samples[len(samples.keys())-1]) != 100:
                # delete last empty trajectory
                del samples[count]

        return {1: samples}
    
    def params_to_list(self, samples: int = 1, option: str = 'weights'):
        """
        converts the params of a model in to a 1D list

        Args:
            option (str): choose to aggregate weights or gradients
        """
        my_list = []
        if option == 'weights':
            for _, param in self.model.named_parameters():
                my_list.extend(
                    param.cpu().detach().reshape(-1).numpy().tolist()
                )
        elif option == 'grad':
            for _, param in self.model.named_parameters():
                my_list.extend(
                    param.grad.cpu().detach().reshape(-1).numpy().tolist()
                )
        
        return my_list
    
    def get_model_shape(self):
        """
        Returns a list which contains for each layer it's shape. 
        The shape is a tuple containing either 1 or 2 elem.
        """
        my_list = []
        for _, param in self.model.named_parameters():
            my_list.append(tuple(param.shape))
        return my_list
    
    def reshape_list(self, shapes, weights):
        """
        Reshapes the list in the number of list that should be set. 
        (This function is hard coded for the current model,
        meaning it won't work if the model changes...)

        Args:
            shapes (list): the shape of each layer of the model (use 
                get_model_shape to get the according shape)
            weights (list): weights of the model in a 1D list format 
                (use tf_weights _to_list to see an example)

        Returns:
            list: the weights rearanged to fit into the tensorflow model
        """
        i=0
        new_list = []
        for elem in shapes:
            if len(elem) == 1:
                new_list.append(np.array(weights[i:i+elem[0]]))
                i += elem[0]
            elif len(elem) == 2:
                temp = []
                for _ in range(elem[0]):
                    temp.append(weights[i:i+elem[1]])
                    i += elem[1]
                new_list.append(np.array(temp))
        return new_list
    
    def set_weights(self, list_of_weights):
        """Sets the weights (list_of_weights) to the current model. 
        The weights are a 1D list (same as the output of weight_to_list)

        Args:
            list_of_weights (list): 1D list of the weights you want to assign to the model.

        """
        shapes = self.get_model_shape()  # [(64, 2), (64, 16), (64,), (64,), (96, 16), (96,)]
        reshaped = self.reshape_list(shapes, list_of_weights)
        # it may does not work for reseting params
        # https://discuss.pytorch.org/t/how-to-assign-an-arbitrary-tensor-to-models-parameter/44082
        # for idx, (_, param) in enumerate(self.model.named_parameters()):
        #     param.data = torch.from_numpy(reshaped[idx]).to(self.device, dtype=torch.float)
        state_dict = self.model.state_dict()
        for idx, (name, _) in enumerate(self.model.named_parameters()):
            state_dict[name] = torch.from_numpy(reshaped[idx]).to(
                self.device, dtype=torch.float
            )
        self.model.load_state_dict(state_dict)
    
    def set_grad(self, list_of_grad):
        """Set the gradients to the current model

        Args:
            list_of_grad (list): 1D list of the gradients
        """
        shapes = self.get_model_shape()
        reshaped = self.reshape_list(shapes, list_of_grad)
        for idx, (name, param) in enumerate(self.model.named_parameters()):
            param.grad = torch.from_numpy(reshaped[idx]).to(
                self.device, dtype=torch.float
            )
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def set_robot_id(self, robotID):
        self.robot_id = robotID

    def set_filename(self, filename):
        self.filename = f"../logs/{filename}"

    def get_model(self):
        return self.model