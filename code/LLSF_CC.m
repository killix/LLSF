
function [ResultAll,predict_target,Outputs] = LLSF_CC(train_data,train_target,test_data,test_target,svm, W_s)
%% LLSF - Classifier Chains
% syntax
%   [ResultAll] = LLSF_CC (train_data,train_target,test_data,test_target,svm)
%
% Input
%   train_data          - training data, num_train x num_dim data matrix
%   train_target        - groundtruth labels of test training data, 
%                         num_label x num_train data matrix
%   test_data           - test data, num_test x num_dim data matrix
%   test_target         - groundtruth labels of test data, num_label x
%                         num_test data matrix
%
% output
%   ResultAll           - all the result for each metric (15) 
%%
    addpath('../evaluation'); addpath('../libsvm'); addpath('..');

    %% Setting Parameters
    switch svm.type
        case 'RBF'
            gamma=num2str(svm.para);
            str=['-t 2 -g ',gamma,' -b 1'];
        case 'Poly'
            gamma=num2str(svm.para(1));
            coef=num2str(svm.para(2));
            degree=num2str(svm.para(3));
            str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
        case 'Linear'
            str='-t 0 -b 1';
        otherwise
            error('SVM types not supported');
    end
    
    
   %% Call the main funciton
    num_test       = size(test_data,1);
    num_class      = size(test_target,1);  
    predict_target = zeros(num_class,num_test); % ×îÖÕ½á¹û
    Outputs        = predict_target;
    Allclassorders = randperm(num_class);

    for i = 1:num_class
        disp(['LLSF-CC: Build Classifier - ',num2str(i),'/',num2str(num_class)]);
        p_index = find(W_s(:,Allclassorders(i))~=0);
        
        if i>1
            traindata = [train_data(:,p_index), train_target(Allclassorders(1:i-1),:)'];
            testdata = [test_data(:,p_index), predict_target(Allclassorders(1:i-1),:)'];
            
        else
            traindata = train_data(:,p_index);
            testdata  = test_data(:,p_index);
        end 
        
        Model = svmtrain(train_target(Allclassorders(i),:)',traindata,str);    
        [Pre_Label,~,Output]=svmpredict(test_target(Allclassorders(i),:)',testdata,Model,'-b 1');
        if(isempty(Pre_Label))
            Pre_Label=train_target(Allclassorders(i),1)*ones(num_test,1);
            if(train_target(Allclassorders(i),1)==1)
                Prob_pos=ones(num_test,1);
            else
                Prob_pos=zeros(num_test,1);
            end
            predict_target(Allclassorders(i),:)=Pre_Label';
            Outputs(Allclassorders(i),:)=Prob_pos';
        else
            pos_index=find(Model.Label==1);
            Prob_pos=Output(:,pos_index);
            predict_target(Allclassorders(i),:)=Pre_Label';
            Outputs(Allclassorders(i),:)=Prob_pos';
        end
    end

   %% Evaluation 
    ResultAll = EvaluationAll(predict_target,Outputs,test_target);
end