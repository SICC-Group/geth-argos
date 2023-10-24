from server import Server
from ai import AI, Model
import os
import time
import torch
import numpy as np
from control_params import params as cp
from aggregation import aggregate

torch.manual_seed(cp['RANDOM_SEED'])
np.random.seed(cp['RANDOM_SEED'])

RESULT_FILE = "logs/training_results.csv"
FILE_VALID_INTERVAL = 15 # 1 = 50s, so 20 is 1000s for the data to be expired.
NUM_OF_ROBOTS = 15
accepted_gradients = []
is_participants = set()
all_samples = dict()

def starting_print():
    print()
    print("#############################################")
    print("## The pytorch server is up and running ##")
    print("#############################################")
    print()

def log(message):
    with open(RESULT_FILE, 'a') as f:
        f.write(message)

def get_filenames(ID):
    path = f"logs/{ID}/"
    all_files = []
    max_file_number = 0
    for files in os.listdir(path):
        number = files[4:-4]
        if number.isdigit():
            number = int(number)
            all_files.append((path+files,number))
            if number>max_file_number:
                max_file_number = number
    return [
        files[0]
        for files in all_files
        if files[1] > max_file_number - FILE_VALID_INTERVAL
    ]

def training_sequence(ai: AI, server: Server, dico: dict):
    colab = dico['colab']
    if colab:
        robot_id = dico['ID']
        weights = dico['weights']
        version = dico['version']
        address = dico['address']
        print(f"training robots {robot_id}'s ({address}) data.")
        filenames = get_filenames(robot_id)

        ai.set_robot_id(robot_id)
        # ai.set_filename(filename)
        ai.set_weights(weights)

        data = ai.load_data(filenames)
        x_train, x_val, y_train, y_val = ai.create_training_and_val_batch(data)
        nb_samples_train = len(x_train)
        train_set, val_set = ai.create_dataset(x_train, x_val, y_train, y_val)
        gradients, loss, iteration, train_time = ai.train_model(
            train_set, steps=nb_samples_train#min(150, nb_samples_train)
        )
        val_loss = ai.val_model(val_set)
        # dist_of_grad(gradients)
        weights, agg_grad = process_grad(
            ai, gradients, robot_id, nb_samples_train
        )  # 1 or 2912

        print(
            "history of robot {}: iter:{}, loss:{}, val_loss:{}, time:{}".format(
                robot_id, iteration, loss, val_loss, train_time
            )
        )
        log("{},{},{},{},{},{},{},{}\n".format(
            iteration, robot_id, address, version, 
            nb_samples_train, loss, val_loss, train_time,
        ))

        print("=========================")
        print("robot_id:", robot_id)
        print("weights:", weights[:1])
        print("gradients:", gradients[:1])
        print("agg_grad: ", agg_grad[:1])
        print("=========================")
    else:
        gradients = dico['grad']
        robot_id = dico['ID']
        weights, agg_grad = process_grad(
            ai, gradients, robot_id, samples=200
        )  # 1 or 2912
        print(f"=== byzantine robot {robot_id} ===")
        print("weight: ", weights[:1])
        print("gradient: ", gradients[:1])
        print("agg_grad: ", agg_grad[:1])
        print()
    
    dico = {
        "nb_samples": 200,
        'gradients': gradients,
        'agg_grad': agg_grad,
        'weights': weights
    }
    server.send_message(dico)  # send to the docker contrainer

def dist_of_grad(gradients):    
    bins = [
        -100, -2, -1, -0.9, -0.8, -0.7, -0.6,
        -0.5, -0.4, -0.3, -0.2, -0.1, 0,
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1, 2, 100
    ]
    hist, _ = np.histogram(gradients, bins)
    hist = hist.tolist()
    formatted_numbers = ['{:<6}'.format(num) for num in hist]
    output = ' '.join(formatted_numbers)
    output = output + f"{min(gradients)}, {max(gradients)} \n"
    with open("/home/fwye/workspace/dist.txt", "a") as file:
        file.write(output)

def process_grad(ai: AI, gradients: list, robot_id: int, samples: int):
    global accepted_gradients, is_participants, all_samples
    accepted_gradients.append(gradients)
    all_samples[robot_id] = samples
    
    # add time
    if len(accepted_gradients) < NUM_OF_ROBOTS:
        return [1], [1]  # 1
    else:
        agg_grad = aggregate(accepted_gradients)
        print(f"\nagg_method: {cp['AGGREGATION']}\n")
        ai.set_grad(agg_grad)  # including gradients step
        weights = ai.params_to_list(option='weights')
        accepted_gradients.clear()
        all_samples.clear()
        return weights, agg_grad


if __name__ == '__main__':
    server = Server(port=9801)
    ai = AI(0, "")

    with open(RESULT_FILE, 'w') as f:
        f.write(
            "iter,robotID,address,version,nb_samples,loss,val_loss,time\n"
        )
    
    # bins = [
    #     -100, -2, -1, -0.9, -0.8, -0.7, -0.6,
    #     -0.5, -0.4, -0.3, -0.2, -0.1, 0,
    #     0.1, 0.2, 0.3, 0.4, 0.5,
    #     0.6, 0.7, 0.8, 0.9, 1, 2, 100
    # ]
    # formatted_numbers = ['{:<6}'.format(num) for num in bins]
    # output = ' '.join(formatted_numbers)
    # with open("/home/fwye/workspace/dist.txt", "w") as file:
    #     file.write(output+"\n")

    starting_print()
    running = True
    while running:
        server.accept_new_connection()
        server.send_message({'info': "ready"})
        dico = server.get_message() # the message the client send is only their robot id.

        if 'weights' in  dico:
            training_sequence(ai, server, dico)
        else: 
            running = False
        # time.sleep(2.5)
    print("Shutting down the Pytorch Server.")
        



