// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    uint8 constant numParticipants = ${MAXWORKERS}; // number of robots in an experiment
    // uint8 constant minParticipant = numParticipants/2; // number of participant required for a round of aggregation
    
    uint16 constant numParam = ${NUMPARAMS}; // number of weights of the neural networks
    uint8 public version;
    uint16 public subVersion;
    int48[numParam] multiKrumW;
    int48[numParam] geoMedW;
    int48[numParam] autoGMW;
    int48[numParam] medianW;
    int48[numParam] trimmedMeanW;
    int48[numParam] centeredClippingW;
    int48[numParam] clusteringW;
    int48[numParam] clippedClusteringW;
    int48[numParam] DnCW;
    int48[numParam] signGuardW;
    int48[numParam] meanW;

    // int48[numParam] multiKrumG;
    // int48[numParam] geoMedG;
    // int48[numParam] autoGMG;
    // int48[numParam] medianG;
    // int48[numParam] trimmedMeanG;
    // int48[numParam] centeredClippingG;
    // int48[numParam] clusteringG;
    // int48[numParam] clippedClusteringG;
    // int48[numParam] DnCG;
    // int48[numParam] signGuardG;
    // int48[numParam] meanG;

    function setInitWeights(int48[numParam] memory weights) external{
        require(version == 0, "The weights have already been initialised.");
        version = 1;
        multiKrumW = weights;
        geoMedW = weights;
        autoGMW = weights;
        medianW = weights;
        trimmedMeanW = weights;
        centeredClippingW = weights;
        clusteringW = weights;
        clippedClusteringW = weights;
        DnCW = weights;
        signGuardW = weights;
        meanW = weights;
    }

    function submultiKrumW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        multiKrumW = weights;
    }
    function getmultiKrumW() external view returns(int48[numParam] memory) {
        return multiKrumW;
    }
    // function submultiKrumG(int48[numParam] memory gradients) external {
    //     multiKrumG = gradients;
    // }
    // function getmultiKrumG() external view returns(int48[numParam] memory) {
    //     return multiKrumG;
    // }

    
    function subgeoMedW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        geoMedW = weights;
    }
    function getgeoMedW() external view returns(int48[numParam] memory) {
        return geoMedW;
    }
    // function subgeoMedG(int48[numParam] memory gradients) external {
    //     geoMedG = gradients;
    // }
    // function getgeoMedG() external view returns(int48[numParam] memory) {
    //     return geoMedG;
    // }


    function subautoGMW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        autoGMW = weights;
    }
    function getautoGMW() external view returns(int48[numParam] memory) {
        return autoGMW;
    }
    // function subautoGMG(int48[numParam] memory gradients) external {
    //     autoGMG = gradients;
    // }
    // function getautoGMG() external view returns(int48[numParam] memory) {
    //     return autoGMG;
    // }


    function submedianW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        medianW = weights;
    }
    function getmedianW() external view returns(int48[numParam] memory) {
        return medianW;
    }
    // function submedianG(int48[numParam] memory gradients) external {
    //     medianG = gradients;
    // }
    // function getmedianG() external view returns(int48[numParam] memory) {
    //     return medianG;
    // }


    function subtrimmedMeanW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        trimmedMeanW = weights;
    }
    function gettrimmedMeanW() external view returns(int48[numParam] memory) {
        return trimmedMeanW;
    }
    // function subtrimmedMeanG(int48[numParam] memory gradients) external {
    //     trimmedMeanG = gradients;
    // }
    // function gettrimmedMeanG() external view returns(int48[numParam] memory) {
    //     return trimmedMeanG;
    // }


    function subcenteredClippingW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        centeredClippingW = weights;
    }
    function getcenteredClippingW() external view returns(int48[numParam] memory) {
        return centeredClippingW;
    }
    // function subcenteredClippingG(int48[numParam] memory gradients) external {
    //     centeredClippingG = gradients;
    // }
    // function getcenteredClippingG() external view returns(int48[numParam] memory) {
    //     return centeredClippingG;
    // }


    function subclusteringW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        clusteringW = weights;
    }
    function getclusteringW() external view returns(int48[numParam] memory) {
        return clusteringW;
    }
    // function subclusteringG(int48[numParam] memory gradients) external {
    //     clusteringG = gradients;
    // }
    // function getclusteringG() external view returns(int48[numParam] memory) {
    //     return clusteringG;
    // }


    function subclippedClusteringW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        clippedClusteringW = weights;
    }
    function getclippedClusteringW() external view returns(int48[numParam] memory) {
        return clippedClusteringW;
    }
    // function subclippedClusteringG(int48[numParam] memory gradients) external {
    //     clippedClusteringG = gradients;
    // }
    // function getclippedClusteringG() external view returns(int48[numParam] memory) {
    //     return clippedClusteringG;
    // }


    function subDnCW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        DnCW = weights;
    }
    function getDnCW() external view returns(int48[numParam] memory) {
        return DnCW;
    }
    // function subDnCG(int48[numParam] memory gradients) external {
    //     DnCG = gradients;
    // }
    // function getDnCG() external view returns(int48[numParam] memory) {
    //     return DnCG;
    // }


    function subsignGuardW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        signGuardW = weights;
    }
    function getsignGuardW() external view returns(int48[numParam] memory) {
        return signGuardW;
    }
    // function subsignGuardG(int48[numParam] memory gradients) external {
    //     signGuardG = gradients;
    // }
    // function getsignGuardG() external view returns(int48[numParam] memory) {
    //     return signGuardG;
    // }


    function submeanW(int48[numParam] memory weights) external {
        subVersion ++ ;
        version = 2 + uint8((subVersion - 1) / 11);
        meanW = weights;
    }
    function getmeanW() external view returns(int48[numParam] memory) {
        return meanW;
    }
    function getmeanW0() external view returns(int48) {
        return meanW[0];
    }
    // function submeanG(int48[numParam] memory gradients) external {
    //     meanG = gradients;
    // }
    // function getmeanG() external view returns(int48[numParam] memory) {
    //     return meanG;
    // }
    

    function getBlockNumber() external view returns (uint) {
        return block.number-1;
    }
    function getAddress() external view returns (address) {
        return msg.sender;
    }
}