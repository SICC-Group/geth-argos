// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    uint8 constant numParticipants = ${MAXWORKERS}; // number of robots in an experiment
    // uint8 constant minParticipant = numParticipants/2; // number of participant required for a round of aggregation
    
    uint16 constant numWeights = ${NUMWEIGHTS}; // number of weights of the neural networks
    int48[numWeights] currentWeights; // the last valid aggregated weights from the previous round
    int48[numWeights] previousWeights;
    int48[numWeights] currentAggGradients;
    int48[numWeights] previousAggGradients;
    // int48[numWeights][numParticipants] acceptedGradients;
    // int48[][] previousGradients;
    address[numParticipants] participantsList; // the participants of a round of aggregation
    address[numParticipants] previousParticipants;
    // event LogEventAfter(address indexed sender, int id, string message);
    // event LogEventBefore(address indexed sender, string message);

    uint8 public version; // the version (the number of aggregation round that took place) of the currentWeights.
    uint8 public currentParticipants;

    struct Participant {
        int samples; // the number of samples associated to the wieghts of this participant
        bool participates; // if he is participating in the current round of aggregation
        int money;
        int robotID;
        int48[numWeights] grad;
    }

    mapping (address => Participant) participantsMap; // mapping from a client address to his personal info.

    /*
    * Payable function used externaly to submit ones weights to the blockchain.
    * For the weights to be elligeable, certain criterion must be followed:
    *   The caller must send aggregationPrice when calling the function. (If his weights are valid
    *   the called will be rewarded minimum this amount.)
    *   He must have trained his weights with more than 0.
    *   He musn't participate multiple times for a same aggregation round.
    * Then if all the above conditions are met, the Weights will be tested against an error function.
    * if the error is greater than errorThreshold, then the weights will be discarded and the sender will loose
    * some ethereum. This is to prevent bad behaving agents.
    * After all this the weights will be processed to take part in the aggregation round.
    *
    * arguments:
    *   nbSamples (int16): The number of samples the weights have been trained on.
    *   weights (int48[numWeights]): List of new weights.  
    */
    function weightsSubmission (
        int48[numWeights] memory weights
    ) external {
        previousWeights = currentWeights;
        currentWeights = weights;
        version ++;
    }

    function aggGradSubmission (
        int48[numWeights] memory gradients
    ) external {
        previousAggGradients = currentAggGradients;
        currentAggGradients = gradients;
    }

    function gradientsSubmission (
        int robot_id,
        int nbSamples
        // int48[numWeights] memory gradients
    ) external {
        // require(
        //     !participantsMap[msg.sender].participates,
        //     "Must not already be in current round of aggregation."
        // );
        // participantsList.push(msg.sender);
        participantsList[currentParticipants] = msg.sender;
        participantsMap[msg.sender].participates = true;
        participantsMap[msg.sender].samples = nbSamples;
        participantsMap[msg.sender].money -= 1;
        participantsMap[msg.sender].robotID = robot_id;
        // participantsMap[msg.sender].grad = gradients;
        // for (uint16 i = 0; i < numWeights; i ++ ) {
        //     acceptedGradients[currentParticipants][i] = gradients[i];
        // }
        currentParticipants ++;

        if (currentParticipants % numParticipants == 0) {
            // loggMessageBefore("OKOKOKOKOKOKOKOK!!");
            prepareNextAggregation();
            // version += 10;
        }
    }

    /*
    * Sets the initial weights, the rankweights and set the version to 1.
    * Can only be called in the beginning of the experiment.
    */
    function setInitWeights(int48[numWeights] memory weights) external{
        require(version == 0, "The weights have already been initialised.");
        currentWeights = weights;
        version ++;
    }

    function prepareNextAggregation() private {
        // new        
        // for (uint8 i = 0; i < participantsList.length; i ++) {
        //     participantsMap[participantsList[i]].participates = false;
        // }
        // previousParticipants = participantsList;
        currentParticipants = 0;
        // delete participantsList;
    }

    // function copyArray(address[] memory source) public {
    //     address[] memory destination = new address[](numParticipants);
    //     for (uint i = 0; i < source.length; i ++ )
    //         destination[i] = source[i];
        // previousParticipants = destination;

        // previousParticipants = source;
    // }


    // //becarfull as there are no floating point values in solidy, the values are normalized such
    // //that the sum of all the values is equal to 10**9.
    // function normalize(int[] memory data, uint length) private pure returns(int[] memory){
    //     int total = 0;
    //     int[] memory normalized = new int[](length);
    //     for (uint i = 0; i < length; i++) {
    //         normalized[i] = data[i] * 10**9;
    //         total = total + data[i];
    //     } 
    //     for (uint i = 0; i < length; i++) normalized[i] = normalized[i]/total;
    //     return normalized;
    // }

    // // returns the median value of a list
    // function median(uint[] memory my_list) public pure returns (uint med){
    //     my_list = sort(my_list);
    //     uint midr = my_list.length / 2;
    //     uint midl = my_list.length - midr -1;
    //     med = (my_list[midl] + my_list[midr])/2;
    // }

    // /* 
    // * Updates the next weights value with the new weights multiplied by the number of samples
    // * they have been trained on.
    // *
    // * arguments:
    // *   nbSamples (int16): The number of samples the weights have been trained on.
    // *   weights (int48[numWeights]): List of new weights.  
    // */ 
    // function initNextWeights(int nbSamples, int48[numWeights] memory weights) private{
    //     for (uint16 i = 0; i < numWeights; i++) nextWeights[i] = weights[i]*int48(nbSamples);
    // }

    // /* 
    // * Increases the next weights value with the new weights multiplied by the number of samples
    // * they have been trained on.
    // *
    // * arguments:
    // *   nbSamples (int16): The number of samples the weights have been trained on.
    // *   weights (int48[numWeights]): List of new weights.  
    // */ 
    // function updateNextWeights(int nbSamples, int48[numWeights] memory weights) private{
    //     for (uint16 i = 0; i < numWeights; i++) nextWeights[i] += weights[i]*int48(nbSamples);
    // }

    /*
    * Updates the currents weights with the last weights sent and the next weights that have been computed overtime.
    *
    * arguments:
    *   nbSamples (int16): The number of samples the weights have been trained on.
    *   weights (int48[numWeights]): List of new weights.  
    */
    // function updateCurrentWeights(int nbSamples, int48[numWeights] memory weights) private{
    //     for (uint16 i = 0; i < numWeights; i++) currentWeights[i] = (nextWeights[i] + weights[i]*int48(nbSamples))/int48(totalSamplesAccepted);
    // }

    // Mean Absolute Error
    function MAE(int48[numWeights] memory weights) public view returns (uint){
        uint mae = 0;
        for (uint16 i = 0; i < numWeights; i++) mae += AE(currentWeights[i], weights[i]);
        return mae/uint(numWeights);
    }

    // Mean Squared Error
    function MSE(int48[numWeights] memory weights) private view returns (uint){
        uint mse = 0;
        for (uint16 i = 0; i < numWeights; i++) mse += SE(currentWeights[i], weights[i]);
        return mse/uint(numWeights);
    }

    // Root Mean Squared Error 
    function RMSE(int48[numWeights] memory weights) private view returns (uint){
        uint rmse = MSE(weights);
        return sqrt(rmse);
    }

    // squared error
    function SE(int48 val1, int48 val2) private pure returns (uint){
        return uint48((val1 - val2)**2);
    }

    // absolute error
    function AE(int48 val1, int48 val2) private pure returns (uint){
        return uint(abs(val1 - val2));
    }

    // absolute function
    function abs(int48 x) private pure returns (int) {
        return x >= 0 ? x : -x;
    }

    // absolute function
    function abs256(int x) private pure returns (int) {
        return x >= 0 ? x : -x;
    }

    // square root function
    function sqrt(uint x) private pure returns (uint y) {
        uint z = (x + 1) / 2;
        y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }

    // function sort(uint[] memory data) private pure returns(uint[] memory) {
    //    quickSort(data, int(0), int(data.length - 1));
    //    return data;
    // }
    
    // function quickSort(uint[] memory arr, int left, int right) private pure{
    //     int i = left;
    //     int j = right;
    //     if(i==j) return;
    //     uint pivot = arr[uint(left + (right - left) / 2)];
    //     while (i <= j) {
    //         while (arr[uint(i)] < pivot) i++;
    //         while (pivot < arr[uint(j)]) j--;
    //         if (i <= j) {
    //             (arr[uint(i)], arr[uint(j)]) = (arr[uint(j)], arr[uint(i)]);
    //             i++;
    //             j--;
    //         }
    //     }
    //     if (left < j)
    //         quickSort(arr, left, j);
    //     if (i < right)
    //         quickSort(arr, i, right);
    // }

  
    // function quickSortComposed(ranker[] memory arr, int left, int right) private pure {
    //     int i = left;
    //     int j = right;
    //     if(i==j) return;
    //     uint pivot = arr[uint(left + (right - left) / 2)].mae;
    //     while (i <= j) {
    //         while (arr[uint(i)].mae < pivot) i++;
    //         while (pivot < arr[uint(j)].mae) j--;
    //         if (i <= j) {
    //             (arr[uint(i)], arr[uint(j)]) = (arr[uint(j)], arr[uint(i)]);
    //             i++;
    //             j--;
    //         }
    //     }
    //     if (left < j)
    //         quickSortComposed(arr, left, j);
    //     if (i < right)
    //         quickSortComposed(arr, i, right);
    // }

    // returns the current weights
    function getWeights() external view returns(int48[numWeights] memory){
        return currentWeights;
    }

    function getAggGradients() external view returns(int48[numWeights] memory){
        return currentAggGradients;
    }

    function getWeights0() external view returns (int48) {
        return currentWeights[0];
    }

    // returns the address of the client making the call.
    function getAddress() external view returns (address) {
        return msg.sender;
    }

    function getParticipantsList() external view returns (address[numParticipants] memory){
        return participantsList;
    }

    function getParticipated() external view returns (bool){
        return  participantsMap[msg.sender].participates;
    }

    function getPreviousParticipants() external view returns (address[numParticipants] memory){
        return previousParticipants;
    }

    // function getPreviousGradientsLen() external view returns (uint) {
    //     return previousGradients.length;
    // }

    function getMoney() external view returns (int){
        return participantsMap[msg.sender].money;
    }
    
    function getBlockNumber() external view returns (uint){
        return block.number-1;
    }

    function getBlockHash() external view returns (bytes32){
        return blockhash(block.number-1);
    }

    function getPreviousBlockHash() external view returns (bytes32){
        return blockhash(block.number-2);
    }

    function getcurrentParticipants() external view returns (uint8) {
        return currentParticipants;
    }

    // function getAcceptedGradientsLen() external view returns (uint) {
    //     return acceptedGradients.length;
    // }

    // function loggMessageBefore(string memory message) public {
    //     emit LogEventBefore(msg.sender, message);
    // }

    // function logMessageAfter(int id, string memory message) public {
    //     emit LogEventAfter(msg.sender, id, message);
    // }
}