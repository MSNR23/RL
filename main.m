clear all;
close all;

%SIM_FILE_NAME = "leg_for_outside_pass";
%run("Leg_TCT_Parameters_ALL6_decFMax.m");
%SIMULATION_FREQ = 1;

%Open the Simulink 🄬 model
mdl = "doublePendulum";
open_system(mdl);

numObs = 5; %number of observation
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = "Angular velocity";

PelvisAngle = 20;

numAct = 1;
actInfo = rlNumericSpec([numAct 1],'LowerLimit',-1,'UpperLimit',1);
actInfo.Name = 'torque';

blk = mdl + '/RLAgent';
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);
env.ResetFcn = @(in) ResetFnc(in);

%Q(S,A|θQ)-Critics that estimate the expected cucmulative long-term reward
%of a policy for a given discrete action A and a given observation S.
obsPath = [...
    featureInputLayer(prod(obsInfo.Dimension)),...
    fullyConnectedLayer(5),...
    reluLayer,...
    fullyConnectedLayer(5,Name="obsout"),...
    ];

actPath = [ %Action path layers
    featureInputLayer(prod(actInfo.Dimension))
    fullyConnectedLayer(5)
    reluLayer
    fullyConnectedLayer(5,Name="actout")
    ];
comPath = [ %Common path to output layers %Concatenate two layers along dimension one
    concatenationLayer(1,2,Name="cct")
    fullyConnectedLayer(5)
    reluLayer
    fullyConnectedLayer(1,Name="output")
];
net = dlnetwork(obsPath);
net = addLayers(net,actPath);
net = addLayers(net,comPath);

net = connectLayers(net,"obsout","cct/in1");
net = connectLayers(net,"actout","cct/in2");
plot(net)

net = initialize(net);
summary(net);

critic = rlQValueFunction(net,obsInfo,actInfo);

actorNet = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(prod(actInfo.Dimension))
    ];
actorNet = dlnetwork(actorNet);
actorNet = initialize(actorNet);
summary(actorNet)
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

criticOpts = rlOptimizerOptions(...
    LearnRate=1e-03,...
    GradientThreshold=1);
actorOpts = rlOptimizerOptions(...
     LearnRate=1e-03,...
     GradientThreshold=1);
opt = rlDDPGAgentOptions(...
    SampleTime=1.0,...
    CriticOptimizerOptions=criticOpts,...
    ActorOptimizerOptions=actorOpts,...
    ExperienceBufferLength=1e5,...
    DiscountFactor=0.99,...
    MiniBatchSize=128);

agent = rlDDPGAgent(actor,critic,opt);

options = trainingOptions("sgdm", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    MaxEpochs=20, ...
    MiniBatchSize=64, ...
    Plots="training-progress")

trainingStats = train(agent,env,rlTrainingOptions);
