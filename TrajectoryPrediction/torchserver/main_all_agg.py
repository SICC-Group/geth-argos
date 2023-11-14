from server import Server
from ai import AI
import os
import time
from copy import deepcopy
import torch
import numpy as np
from control_params import params as cp
from aggregation import aggregate
from adversary import attack

# work with all aggregation

torch.manual_seed(cp['RANDOM_SEED'])
np.random.seed(cp['RANDOM_SEED'])

RESULT_FILE = "logs/training_results_"
FILE_VALID_INTERVAL = 15 # 1 = 50s, so 20 is 1000s for the data to be expired.
NUM_OF_ROBOTS = 15
AGGREGATION_NAMES = [
    "multiKrum", "geoMed", "autoGM", "median",
    "trimmedMean", "centeredClipping", "clustering",
    "clippedClustering","DnC", "signGuard", "mean"
]
benign_gradients = {name: [] for name in AGGREGATION_NAMES}
byzantine_gradients = {name: [] for name in AGGREGATION_NAMES}
last_round_grad = {name: [0] * 2912 for name in AGGREGATION_NAMES}
aggregated_gradients = {}
updated_weights = {}

def starting_print():
    print()
    print("#############################################")
    print("## The pytorch server is up and running ##")
    print("#############################################")
    print()

def log(filename, message):
    with open(filename, 'a') as f:
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

def benign_train(ai: AI, dico: dict):
    global all_weights
    robot_id = dico['ID']
    all_weights = dico['all_weights']
    version = dico['version']
    address = dico['address']
    print(f"training robots {robot_id}'s ({address}) data.")
    print(f"version: {version}")
    filenames = get_filenames(robot_id)

    ai.set_robot_id(robot_id)
    # ai.set_filename(filename)
    data = ai.load_data(filenames)
    x_train, x_val, y_train, y_val = ai.create_training_and_val_batch(data)
    nb_samples_train = len(x_train)
    train_set, val_set = ai.create_dataset(x_train, x_val, y_train, y_val)

    start = time.time()
    for aggregation_name in AGGREGATION_NAMES:
        weights = all_weights[aggregation_name]
        print(f"{aggregation_name} weight 0: {weights[0]}")
        ai.set_weights(weights)

        gradients, loss, iteration, train_time = ai.train_model(
            train_set, steps=nb_samples_train # min(150, nb_samples_train)
        )
        iteration = (iteration - 1) // len(AGGREGATION_NAMES) + 1
        val_loss = ai.val_model(val_set)
        print(
            "history of robot {}: iter:{}, loss:{:.10f}, val_loss:{:.10f}, "
            "time:{:.5f}, aggregation:{}".format(
                robot_id, iteration, loss, val_loss,
                train_time, aggregation_name
            )
        )
        log(
            RESULT_FILE + aggregation_name + ".csv",
            "{},{},{},{},{},{},{},{}\n".format(
                iteration, robot_id, address, version, 
                nb_samples_train, loss, val_loss, train_time
            )
        )

        new_weights, agg_grad = process_grad(
            ai, gradients, aggregation_name, is_colab=True
        )
        if len(new_weights) > 1 and len(agg_grad) > 1:
            updated_weights[aggregation_name] = new_weights
            aggregated_gradients[aggregation_name] = agg_grad
    
    test = time.time() - start
    print("\nrobot {} has finished".format(robot_id))
    print("time: ===={:.17f}====".format(test))

def byzantine_attack(ai: AI, dico: dict):
    global last_round_grad, benign_gradients
    adversary = dico["adversary"]
    robot_id = dico['ID']
    print(f"===== byzantine robot {robot_id} =====")

    for aggregation_name in AGGREGATION_NAMES:
        num_of_byzantine = (
            NUM_OF_ROBOTS - len(benign_gradients[aggregation_name])
        )
        gradients = attack(
            benign_gradients=benign_gradients[aggregation_name],
            num_of_byzantine=num_of_byzantine,
            adversary=adversary,
            aggregation=aggregation_name,
            num_of_clients=NUM_OF_ROBOTS,
            last_round=last_round_grad[aggregation_name],
        )
        print("benign robots: {}; {} adversary "
              "gradient0: [{:.10f}], in {}".format(
            len(benign_gradients[aggregation_name]), adversary,
            gradients[0], aggregation_name, 
        ))
        new_weights, agg_grad = process_grad(
            ai, gradients, aggregation_name, is_colab=False
        )
        if len(new_weights) > 1 and len(agg_grad) > 1:
            updated_weights[aggregation_name] = new_weights
            aggregated_gradients[aggregation_name] = agg_grad
    print("robot {} has finished".format(robot_id))

def process_grad(ai: AI, gradients: list, agg: str, is_colab: bool):
    global benign_gradients, byzantine_gradients, all_weights
    if is_colab:
        benign_gradients[agg].append(gradients)
    else:
        byzantine_gradients[agg].append(gradients)
    
    if (len(benign_gradients[agg]) + len(byzantine_gradients[agg]) ==
        NUM_OF_ROBOTS):
        print("{} benign workers: , {} byzantine workers".format(
            len(benign_gradients[agg]), len(byzantine_gradients[agg])
        ))
        agg_grad = aggregate(
            benign_gradients[agg],
            byzantine_gradients[agg],
            method=agg
        )
        print(f"aggregation {agg} has finished")
        print("aggregated gradients 0: [{:.10f}] in {}".format(
            agg_grad[0], agg
        ))
        ai.set_weights(all_weights[agg])
        ai.set_grad_step(agg_grad)  # including gradients step
        weights = ai.params_to_list(option='weights')
        print("new weight 0: [{:.10f}]\n".format(weights[0]))
        benign_gradients[agg].clear()
        byzantine_gradients[agg].clear()
        return weights, agg_grad
    else:
        return [1], [1]

def training_sequence(ai: AI, server: Server, dico: dict):
    global last_round_grad
    colab = dico['colab']
    if colab:
        benign_train(ai, dico)
    else:
        byzantine_attack(ai, dico)
    
    if (len(updated_weights) == len(AGGREGATION_NAMES) and 
        len(aggregated_gradients) == len(AGGREGATION_NAMES)):
        last_round_grad = deepcopy(aggregated_gradients)
        dico = {
            'aggregated_gradients': aggregated_gradients,
            'updated_weights': updated_weights
        }
        server.send_message(dico)  # send to the docker contrainer
        updated_weights.clear()
        aggregated_gradients.clear()
    else:
        dico = {
            'aggregated_gradients': [1],
            'updated_weights': [1]
        }
        server.send_message(dico)  # send to the docker contrainer
    
    print("=================================")
    print("=================================\n\n")


if __name__ == '__main__':
    server = Server(port=9801)
    ai = AI(0, "")

    for aggregation_name in AGGREGATION_NAMES:
        with open(RESULT_FILE + aggregation_name + ".csv", 'w') as f:
            f.write(
                "iter,robotID,address,version,nb_samples,loss,val_loss,time\n"
            )
    
    starting_print()
    running = True
    while running:
        server.accept_new_connection()
        server.send_message({'info': "ready"})
        dico = server.get_message() # the message the client send is only their robot id.

        if 'colab' in dico:
            training_sequence(ai, server, dico)
        else: 
            running = False
        # time.sleep(2.5)
    print("Shutting down the Pytorch Server.")
        



