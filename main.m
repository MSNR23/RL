clear all;
close all;

%SIM_FILE_NAME = "leg_for_outside_pass";
%run("Leg_TCT_Parameters_ALL6_decFMax.m");
%SIMULATION_FREQ = 1;

%Open the Simulink 🄬 model
mdl = "doublePendulum";
open_system(mdl);

numObs = 1; %number of observation
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = "Angular velocity";

numAct = 1;
actInfo = rlNumericSpec([numAct 1],'LowerLimit',-1,'UpperLimit',1);
actInfo.Name = 'torque';

blk = mdl + '/RL Agent';
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);
env.ResetFcn = @ResetFnc(PelvisAngle);

%Q(S,A|θQ)-Critics that estimate the expected cumulative long-term reward
%of a policy for a given discrete action A and a given observation S.
obsPath = [ %Observation path layers
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(5)
    reluLayer
    fullyConnectedLayer(5,Name="obsout")
    ];
actPath = [ %Action path layers
    featureInputLayer(prod(actInfo.Dimension))
    fullyConnectedLayer(5)
    reluLayer
    fullyConnectedLayer(5,Name="actout")
    ];
comPath = [ %Common path to output layers %Concatenate two layers along dimension one
    concatenationLayer(1,2,Name="cct")
    fullyConnectedLayer
    reluLayer
    fullyConnectedLayer(1,Name="output")
];
net = dlnetwork;
net = addLayers(net,obsPath);
net = addLayers(net,actPath);
net = addLayers(net,comPath);

net = connectLayers(net,"obsout","cct/in1");
net = connectLayers(net,"actout","cct/in2");
plot(net)

net = initialize(net);
summary(net);

critic = rlQValueFunction(net,observationInfo,actorInfo);

opt = rlQAgentOptions(SampleTime=1.0);
opt.DiscountFacor = 0.95;

agent = rlQAgent(critic,agentOptions);

trainingStats = train(agent,env,trainOpts);
