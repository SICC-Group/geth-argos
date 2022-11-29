// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract MarketForaging {

  int[4] Demand               = [int(0), 0, -1, 6];
  uint constant quota         = QUOTA;
  uint constant fuel_cost     = FUELCOST;
  uint constant max_workers   = MAXWORKERS;

  struct Robot {
    bool isRegistered;
    uint efficiency;
    uint income;
    uint balance;
    uint task;
  }

  struct Epoch {
    uint number;
    uint start;
    uint supply;
    uint consumption;
    uint[] C; 
    uint[] MC; 
    uint TC;
    uint ATC;
  }

  struct Patch {

    // Details 
    int x;
    int y;
    uint qtty;
    uint util;
    string qlty;
    string json; 

    // Identifier
    uint id;

    // Assignment
    uint max_workers;
    uint tot_workers;

    Epoch epoch;
  } 

  Patch[] private patches;
  Epoch[] public  epochs;

  uint id_nonce;

  mapping(address => uint) drops;
  mapping(address => uint) lastD;
  mapping(address => Robot) public robot;

  function register(uint eff) public {
    if (!robot[msg.sender].isRegistered){
      robot[msg.sender].isRegistered = true;
      robot[msg.sender].efficiency   = eff;
      robot[msg.sender].income       = 0;
      robot[msg.sender].balance      = 20000;
    }
  }

  function updatePatch(int _x, int _y, uint _qtty, uint _util, string memory _qlty, string memory _json) public {

    uint i = findByPos(_x, _y);

    // If patch is already known
    if (i < 9999) { 
      patches[i].qtty  = _qtty;
      patches[i].util  = _util;
      patches[i].qlty  = _qlty;
      patches[i].json  = _json;
    }

    // If patch is new
    else {

      // Increment unique patch id
      id_nonce++;

      // Append new patch to list
      Epoch memory new_epoch = Epoch({
                          number: 1, 
                          start: block.number, 
                          supply: 0,
                          consumption: 0,
                          C: new uint[](0),
                          MC: new uint[](0),
                          TC: 0,
                          ATC: 0
                        }); 

      patches.push(Patch({
                          x: _x, 
                          y: _y, 
                          qtty: _qtty, 
                          util: _util,
                          qlty: _qlty, 
                          json: _json,
                          id:      id_nonce,
                          max_workers:   max_workers,
                          tot_workers:   0,
                          epoch: new_epoch
                        }));
    }
  } 

  function buyFuel(uint duration) internal {

    // uint tokens_consumed = robot[msg.sender].efficiency * duration * fuel_cost / 10000;
    uint tokens_consumed = duration;

    // efficiency:  amps/block     -- robot internal parameter
    // duration:    centi-seconds  -- duration of trip
    // fuel_cost:   tokens/amp     -- cost of 1 amp

    if (robot[msg.sender].balance < tokens_consumed) {
      robot[msg.sender].balance = 0;
    }
    else {
      robot[msg.sender].balance -= tokens_consumed;
    }
    
  }

  function dropResource(int _x, int _y, uint _qtty, uint _util, string memory _qlty, string memory _json, uint _Q, uint _TC) public {

    uint i = findByPos(_x, _y);

    if (i < 9999 && robot[msg.sender].task == patches[i].id) {

      // New good in current epoch
      patches[i].epoch.supply++;
    
      // Update patch information
      updatePatch(_x, _y, _qtty, _util, _qlty, _json);

      // Pay the robot
      robot[msg.sender].income  += _Q*patches[i].util;
      robot[msg.sender].balance += _Q*1000*patches[i].util;

      // Buy the fuel and update firm analysis
      uint duration = (block.number - patches[i].epoch.start);
      patches[i].epoch.C.push(duration);

      if (patches[i].epoch.C.length > 1) {
        uint new_mc = duration-patches[i].epoch.C[patches[i].epoch.C.length-1];
        patches[i].epoch.MC.push(new_mc);
      }

      buyFuel(_TC);

      if (patches[i].epoch.supply == patches[i].tot_workers) {

        // Finalize previous epoch
        patches[i].epoch.TC  = duration;
        patches[i].epoch.ATC = 100*duration/patches[i].epoch.supply;
        patches[i].epoch.consumption = 1000*patches[i].epoch.supply/patches[i].epoch.TC;


        // // Rules for increasing number of workers
        // if (patches[i].epoch.consumption < quota) {
        //   patches[i].max_workers++;
        // }

        // Init new epoch
        epochs.push(patches[i].epoch);
        patches[i].epoch.number++;
        patches[i].epoch.start = block.number+5;
        patches[i].epoch.supply = 0;
        patches[i].epoch.TC = 0;
        patches[i].epoch.C = new uint[](0);
        patches[i].epoch.MC = new uint[](0);

        // patches[i].epoch.duration
        // patches[i].epoch.consumption

      }
    }
  }

  function assignToPatch(uint i) public {

    // Assign robot to chosen patch
    if (patches[i].tot_workers < patches[i].max_workers) {
      patches[i].tot_workers++;
      robot[msg.sender].task = patches[i].id;
    }
  }

  function assignPatch() public {

    // Limit foragers algorithm
    uint i = findAvailiable();

    // Assign new foraging task
    if (i < patches.length) {
      patches[i].tot_workers++;
      robot[msg.sender].task = patches[i].id;
    }
  }

  function findAvailiable() private view returns (uint) {

    uint index = 9999;

    for (uint i=0; i < patches.length; i++) {
      if (patches[i].tot_workers < patches[i].max_workers && robot[msg.sender].task != patches[i].id) {
        index = i;
      }
    }
      return index;    
  }

  function findByID(uint id) private view returns (uint) {
    for (uint i=0; i < patches.length; i++) {
      if (patches[i].id == id) {
        return i;
      }
    } 
    return 9999;
  }

  function findByPos(int _x, int _y) private view returns (uint) {
    for (uint i=0; i < patches.length; i++) {
      if (_x == patches[i].x && _y == patches[i].y) {
        return i;
      }
    } 
    return 9999;
  }

  function getPatches() public view returns (Patch[] memory){
    return patches;
  }

  function getEpochs() public view returns (Epoch[] memory){
    return epochs;
  }

  function getPatch() public view returns (Patch memory){

    for (uint i=0; i < patches.length; i++) {
      if (patches[i].id == robot[msg.sender].task) return patches[i];
    }   
  }

  function getMyPatch() public view returns (string memory){

    uint task_id = robot[msg.sender].task;

    if (task_id == 0) return "";

    for (uint i=0; i < patches.length; i++) {
      if (patches[i].id == task_id) return patches[i].json;
    } 
  }

  function getBalance() public view returns (uint){
    return robot[msg.sender].balance;
  }
}
  

// function runningMean(uint previous, uint current, uint N) private pure returns (uint) {
//   return (previous*N + current) / (N+1);
//   }



  // function random(uint mod) private view returns (uint) {
  //   return uint(keccak256(abi.encode(block.timestamp))) % mod;
  // }

  // function coinFlip(uint odds) private view returns (bool) {
  //   if (random(100) < odds) {
  //     return true;
  //   }
  //   return false;
  // }




// Backups
// && patches[i].qtty > patches[i].worker_count